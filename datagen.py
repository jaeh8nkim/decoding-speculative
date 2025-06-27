"""
python datagen.py 2 0 111:121,131,134:173
python datagen.py 3 1 250:499

  7B, fp16:  14 GB + kv cache 8192:   ~4 GB per sampling = ~22 GB (~0.50)
0.6B, fp16: 1.2 GB + kv cache 8192: ~0.8 GB per sampling =  ~3 GB (~0.10)
"""

import sys
from tqdm import tqdm

if len(sys.argv) != 4:
    print("Usage: python datagen.py <LARGE_GPU_INDEX> <SMALL_GPU_INDEX> <DF_INDEX_STRING>")
    sys.exit(1)

LARGE_GPU_INDEX = sys.argv[1]
SMALL_GPU_INDEX = sys.argv[2]
DF_INDEX_STRING = sys.argv[3]

def parse_indices(index_string):
    """Parse index string like '123:125,129,140:145' into list of indices"""
    indices = []
    parts = index_string.split(',')
    
    for part in parts:
        part = part.strip()
        if ':' in part:
            # Range like "123:125" (both inclusive)
            start, end = part.split(':')
            start, end = int(start), int(end)
            indices.extend(range(start, end + 1))
        else:
            # Single index like "129"
            indices.append(int(part))
    
    return sorted(list(set(indices)))  # Remove duplicates and sort

try:
    TARGET_INDICES = parse_indices(DF_INDEX_STRING)
    print(f"Processing indices: {TARGET_INDICES}")
except ValueError as e:
    print(f"Error parsing index string '{DF_INDEX_STRING}': {e}")
    sys.exit(1)

# Reverse Speculative Decoding with vLLM in Token ID Space
#
# - Large model generates candidate tokens, small model validates them based on:
#   * Token must be in small model's top-20 predictions
#   * Token probability must exceed 0.01 threshold in small model's token distribution
# - Operates entirely in token ID space using vLLM's TokensPrompt to avoid Unicode issues (U+FFFD)
#   that commonly occur with rare mathematical symbols in reasoning traces when using BPE tokenization
# - Vocabulary matching mask ensures only compatible tokens are proposed
# - Special token mapping handles tokens unique to the small model's vocabulary by translating
#   them to equivalent token sequences for the large model

# --------------------------- imports ---------------------------------------
import os, html, uuid, asyncio, contextlib, nest_asyncio, logging
from IPython.display import HTML, display

import torch
from huggingface_hub import snapshot_download
from vllm import TokensPrompt  # CHANGE: Added TokensPrompt import
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams, RequestOutputKind

nest_asyncio.apply()
torch.set_grad_enabled(False)
logging.disable(logging.INFO)

# --------------------------- configuration ---------------------------------
LARGE_MODEL_NAME  = "simplescaling/s1.1-7B"
SMALL_MODEL_NAME  = "Qwen/Qwen3-0.6B"

LARGE_TEMPERATURE = 0.7
SMALL_TEMPERATURE = 0.7
MAX_SEQ_LEN       = 8192
MAX_NEW_TOKENS    = MAX_SEQ_LEN - 1024

# ---------------- utility: temporarily set visible GPUs --------------------
@contextlib.contextmanager
def visible_gpus(devices: str):
    original = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    print(f"\nCUDA_VISIBLE_DEVICES = {devices}")
    try:
        yield
    finally:
        os.environ["CUDA_VISIBLE_DEVICES"] = original

# --------------------------- engine setup ----------------------------------
async def setup_engines():
    global large_engine, small_engine, large_tokenizer, small_tokenizer
    global large_vocab_size, small_vocab_size, vocab_match_mask, token_mapping
    
    large_checkpoint = snapshot_download(LARGE_MODEL_NAME)
    small_checkpoint = snapshot_download(SMALL_MODEL_NAME)

    with visible_gpus(LARGE_GPU_INDEX):
        print("torch sees", torch.cuda.device_count(), "GPU(s)")              
        large_engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(model=large_checkpoint, 
                            tensor_parallel_size=1,
                            max_model_len=MAX_SEQ_LEN, 
                            gpu_memory_utilization=0.60,
                            dtype="float16"),
            start_engine_loop=True)
        
        large_tokenizer = await large_engine.get_tokenizer()

    with visible_gpus(SMALL_GPU_INDEX):
        print("torch sees", torch.cuda.device_count(), "GPU(s)")              
        small_engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(model=small_checkpoint, 
                            tensor_parallel_size=1,
                            max_model_len=MAX_SEQ_LEN, 
                            gpu_memory_utilization=0.20,
                            dtype="bfloat16"),
            start_engine_loop=True)
        
        small_tokenizer = await small_engine.get_tokenizer()

    # Get model configs using async methods
    large_model_config = await large_engine.get_model_config()
    small_model_config = await small_engine.get_model_config()
    
    large_vocab_size = large_model_config.get_vocab_size()
    small_vocab_size = small_model_config.get_vocab_size()
    
    print(f"Large vocab size: {large_vocab_size}")
    print(f"Small vocab size: {small_vocab_size}")
    print(f"Difference      : {abs(large_vocab_size - small_vocab_size)}")

    vocab_match_mask = torch.zeros(max(large_vocab_size, small_vocab_size), dtype=torch.float32)
    mismatches = []

    for idx in range(min(large_vocab_size, small_vocab_size)):
        large_token = large_tokenizer.convert_ids_to_tokens(idx)
        small_token = small_tokenizer.convert_ids_to_tokens(idx)
        
        if large_token == small_token:
            vocab_match_mask[idx] = 1.0
        else:
            mismatches.append((idx, large_token, small_token))

    print(f"Unmatched tokens: {len(mismatches)}")

    # print every mismatch 
    print(f"\n{'ID':>6}  {'Large token':<25}  Small token")
    for idx, large_token, small_token in mismatches:
        large_token  = "None" if large_token is None else str(large_token)
        small_token  = "None" if small_token is None else str(small_token)
        print(f"{idx:6}  {large_token:<25}  {small_token}")

    # show how the large tokenizer splits "<think>" and "</think>"
    for token_str in ("<think>", "</think>"):
        token_ids   = large_tokenizer.encode(token_str, add_special_tokens=False)
        token_pieces = [large_tokenizer.convert_ids_to_tokens(token_id) for token_id in token_ids]

        print(f"\nTokenization of {token_str!r} by the large tokenizer:")
        print(f"{'ID':>6}  Token piece")
        for token_id, token_piece in zip(token_ids, token_pieces):
            print(f"{token_id:6}  {token_piece}")

    # U+FFFD FIX: Added token mapping for special tokens
    # Create mapping for tokens that only exist in small model
    # When small model generates these tokens, we need to translate them
    # to equivalent token sequences for the large model
    token_mapping = {
        151665: [27, 14172, 9655, 29],   # <tool_response>  --> <, tool, _response, >
        151666: [522, 14172, 9655, 29],  # </tool_response> --> </, tool, _response, >
        151667: [13708, 766, 29],        # <think>          --> <th, ink, >
        151668: [522, 26865, 29],        # </think>         --> </, think, >
    }

# --------------------------- sampling params -------------------------------
large_sampling_params = SamplingParams(
    max_tokens  = 1,
    temperature = LARGE_TEMPERATURE,
    top_p       = 0.95, 
    logprobs    = 20,
    output_kind = RequestOutputKind.DELTA,
)
small_sampling_params = SamplingParams(
    max_tokens  = 1,
    temperature = SMALL_TEMPERATURE,
    top_p       = 0.95, 
    logprobs    = 20,
    output_kind = RequestOutputKind.DELTA,
)

# ------------------------- core decode loop --------------------------------
# U+FFFD FIX: Modified one_step to accept token IDs and use TokensPrompt
async def one_step(engine, sampling_params, context_ids):
    # Pass token IDs directly to vLLM using TokensPrompt
    # This avoids any decoding issues with partial tokens
    tokens_prompt = TokensPrompt(prompt_token_ids=context_ids)
    generator = engine.generate(tokens_prompt, sampling_params, request_id=str(uuid.uuid4()))
    return (await anext(generator)).outputs[0]

async def mixed_decode(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS):
    context_ids_small = small_tokenizer.encode(prompt)
    context_ids_large = []
    
    # For large model, we need to handle special tokens during initial tokenization
    for token_id in context_ids_small:
        if token_id in token_mapping:
            context_ids_large.extend(token_mapping[token_id])
        else:
            context_ids_large.append(token_id)
    
    step_index = 0
    PROB_THRESHOLD = 0.01  
    NUM_TRIALS = 5 

    # Create tqdm progress bar
    pbar = tqdm(total=max_new_tokens, desc="Generating tokens", unit="tok")

    for _ in range(max_new_tokens):
        large_output, small_output = await asyncio.gather(
            one_step(large_engine, large_sampling_params, context_ids_large),
            one_step(small_engine, small_sampling_params, context_ids_small))

        # Extract probabilities from large model output - logprobs is a list
        large_logprobs_dict = large_output.logprobs[0]  
        large_probs = {}
        for token_id, logprob in large_logprobs_dict.items():
            if vocab_match_mask[token_id] > 0:  # Only include vocab-matched tokens
                large_probs[token_id] = torch.exp(torch.tensor(logprob.logprob)).item()
        
        idx_pool = torch.tensor(list(large_probs.keys()))
        prob_pool = torch.tensor(list(large_probs.values()))
        prob_pool = prob_pool / prob_pool.sum()  # Normalize for sampling

        # Extract probabilities from small model output - compact dict for lookup  
        small_logprobs_dict = small_output.logprobs[0] 
        small_probs = {}
        for token_id, logprob in small_logprobs_dict.items():
            small_probs[token_id] = torch.exp(torch.tensor(logprob.logprob)).item()

        # Try to accept a token from large model's distribution
        fallback = True
        for _ in range(NUM_TRIALS):
            pool_idx = torch.multinomial(prob_pool, 1).item()
            candidate_token_id = idx_pool[pool_idx].item()
            if candidate_token_id in small_probs and small_probs[candidate_token_id] >= PROB_THRESHOLD:
                chosen_id = candidate_token_id
                fallback = False
                break

        # Fallback: sample from small model if no acceptance
        if fallback:
            idx_pool = torch.tensor(list(small_probs.keys()))
            prob_pool = torch.tensor(list(small_probs.values()))
            prob_pool = prob_pool / prob_pool.sum()
            pool_idx = torch.multinomial(prob_pool, 1).item()
            chosen_id = idx_pool[pool_idx].item()

        # Get token text for display only
        chosen_text = small_tokenizer.decode([chosen_id])
        large_probability = large_probs.get(chosen_id, 0.0)
        small_probability = small_probs.get(chosen_id, 0.0)

        step_index += 1
        record = dict(
            idx=step_index, 
            text=chosen_text,  # For display only
            token_id=chosen_id,
            fallback=fallback, 
            large_probability=large_probability, 
            small_probability=small_probability,
        )
        yield record

        # print(f"{step_index:4d}{'*' if fallback else ' '}\t"
        #       f"{large_probability:.4f}\t{small_probability:.4f}\t"
        #       f"{chosen_id}\t'{chosen_text}'",
        #       flush=True)

        # Update progress bar
        pbar.update(1)

        # Append to context in ID space for both models
        context_ids_small.append(chosen_id)
        
        # For large model, check if we need to map the token
        if chosen_id in token_mapping:
            # Append the mapped token sequence
            context_ids_large.extend(token_mapping[chosen_id])
        else:
            # Regular token, just append
            context_ids_large.append(chosen_id)
        
        if chosen_id == small_tokenizer.eos_token_id:
            break
    
    pbar.close()

# ---------------------- high-level convenience -----------------------------
async def run_mixed_decode(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS):

    records = []
    
    async for record in mixed_decode(prompt, max_new_tokens):
        records.append(record)
    
    return records

# ------------------------ fire up the engines ------------------------------
# await setup_engines()

# --------------------------- example usage ---------------------------------
import re, openai, sqlite3, time, random
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def update_entry_trace(db_path, index, new_trace_value):
    """Update a single entry's trace field in SQLite with retry logic"""
    max_retries = 5
    
    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect(db_path)
            # Enable WAL mode and set timeout for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")  # 30 seconds
            cursor = conn.cursor()
            
            # Check if entry exists and get current trace
            cursor.execute("SELECT trace FROM dataset WHERE rowid = ?", (index + 1,))
            result = cursor.fetchone()
            
            if not result:
                print(f"Error: Index {index} not found in dataset")
                conn.close()
                return False
            
            current_trace = result[0]
            if current_trace and str(current_trace).strip():
                print(f"Entry {index} already has a trace, skipping")
                conn.close()
                return True
            
            # Update just this row
            cursor.execute("UPDATE dataset SET trace = ? WHERE rowid = ?", 
                          (new_trace_value, index + 1))
            conn.commit()
            conn.close()
            
            print(f"Successfully updated entry {index}")
            return True
            
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Database locked on write for entry {index}, retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"Failed to update entry {index} after {attempt + 1} attempts: {e}")
                return False
        except Exception as e:
            print(f"Failed to update entry {index}: {e}")
            return False

def read_entry_trace(db_path, index):
    """Read the trace value for a specific entry with retry logic"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")  # 5 seconds for writes
            cursor = conn.cursor()
            cursor.execute("SELECT trace FROM dataset WHERE rowid = ?", (index + 1,))
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                return None
            
            trace = result[0]
            return trace if trace and str(trace).strip() else None
            
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and attempt < max_retries - 1:
                wait_time = 0.5 + random.uniform(0, 0.5)
                time.sleep(wait_time)
                continue
            else:
                print(f"Error reading entry {index} after {attempt + 1} attempts: {e}")
                return None
        except Exception as e:
            print(f"Error reading entry {index}: {e}")
            return None

def read_entry_qanda(db_path, index):
    """Read question and answer for a specific entry with retry logic"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=2000")  # 2 seconds for reads
            cursor = conn.cursor()
            cursor.execute("SELECT question, answer FROM dataset WHERE rowid = ?", (index + 1,))
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                return None, None
            
            return result[0], result[1]
            
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and attempt < max_retries - 1:
                wait_time = 0.5 + random.uniform(0, 0.5)
                time.sleep(wait_time)
                continue
            else:
                print(f"Error reading Q&A for entry {index} after {attempt + 1} attempts: {e}")
                return None, None
        except Exception as e:
            print(f"Error reading entry {index}: {e}")
            return None, None

# U+FFFD FIX: Modified to decode entire ID sequence at once
def extract_boxed_answer(records, tokenizer):
    """Extract the last \\boxed{} answer between tokens 151668 and 151645"""
    token_ids = [record['token_id'] for record in records]
    
    # Find positions of the tokens
    pos_151668 = [i for i, tid in enumerate(token_ids) if tid == 151668]
    pos_151645 = [i for i, tid in enumerate(token_ids) if tid == 151645]
    
    if len(pos_151668) != 1 or len(pos_151645) == 0:
        return None

    start_pos = pos_151668[0]
    end_pos = pos_151645[0]  # Take the first occurrence of 151645
    
    if start_pos >= end_pos:
        return None

    # Extract token IDs between the markers (including the end marker)
    between_token_ids = token_ids[start_pos:end_pos+1]
    
    # Decode the entire sequence at once to avoid U+FFFD issues
    between_text = tokenizer.decode(between_token_ids)
    
    # Find all \\boxed{} patterns with proper brace matching
    matches = []
    i = 0
    while i < len(between_text):
        boxed_start = between_text.find('\\boxed{', i)
        if boxed_start == -1:
            break
        
        j = boxed_start + 7  # Start after '\\boxed{'
        brace_count = 1
        while j < len(between_text) and brace_count > 0:
            if between_text[j] == '{':
                brace_count += 1
            elif between_text[j] == '}':
                brace_count -= 1
            j += 1
        
        if brace_count == 0:
            matches.append(between_text[boxed_start + 7:j-1])
        
        i = boxed_start + 1
    
    return matches[-1] if matches else None

def llm_grader(expected_answer, boxed_answer, openai_client, model_name="gpt-4o-mini"):

    def grader_prompt(expected_answer, boxed_answer):
        """Creates the system and user prompts for grading."""
        system_prompt = (
            f"You are an expert grader tasked with evaluating the correctness of an answer.\n"
            f"You will be provided with two pieces of text: the expected answer and the generated answer.\n"
            f"Your task is to determine if the generated answer is semantically equivalent to the expected answer.\n"
            f"Ignore minor formatting differences, extra whitespace, or trivial variations. For numerical answers, consider equivalent representations as correct (e.g., '1/2' and '0.5').\n"
            f"Respond with exactly one word: either 'true' (if correct) or 'false' (if incorrect). Do not include quotation marks, explanations, or any other text.\n"
        )
        user_prompt = (
            f"Expected answer:\n"
            f"{expected_answer}\n"
            f"Generated answer:\n"
            f"{boxed_answer}\n"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return messages
    
    def grader(grading_messages, openai_client, model_name):
        api_response = openai_client.chat.completions.create(
            model=model_name,
            messages=grading_messages
        ).choices[0].message.content
        
        grade = api_response.strip().lower()
        return grade
    
    grading_messages = grader_prompt(expected_answer, boxed_answer)
    grade = grader(grading_messages, openai_client, model_name)
    
    # Ensure the grade is exactly 'true' or 'false'
    if grade in ['true', 'false']:
        return grade
    else:
        # Fallback in case the API returns something unexpected
        return 'false'

async def main():
    NUM_SAMPLINGS = 16

    # Fire up the engines
    await setup_engines()
    
    dataset_file = 'dataset_4qwen3.db'
    good_traces_count = 0

    for i in TARGET_INDICES:
        # Check if entry already has a trace (thread-safe)
        existing_trace = read_entry_trace(dataset_file, i)
        
        if existing_trace:
            good_traces_count += 1
            print(f"Entry {i} already has a trace, skipping")
            continue
            
        # Get question and answer for this entry
        question, expected_answer = read_entry_qanda(dataset_file, i)
        
        system_prompt = (
            f"You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n"
            f"You must respond to every query in the following manner:\n"
            f"First, provide a step-by-step logical exploration of the problem.\n"
            f"Then, provide a clear and direct response based on your reasoning, with the final answer enclosed in \\boxed{{}}."
        )

        input = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>"
        )

        candidate_traces = []
        
        for j in range(NUM_SAMPLINGS):
            print(f"Started working on entry {i}, trial {j}")

            records = await run_mixed_decode(input)
            
            # First check: Token count
            print(f"Token count: {len(records)}")
            
            # Second check: Token validation and fallback rate
            token_ids = [record['token_id'] for record in records]
            count_151668 = token_ids.count(151668)
            contains_151645 = 151645 in token_ids
            
            # Calculate fallback rate
            fallback_count = sum(record['fallback'] for record in records)
            fallback_rate = (fallback_count / len(records)) * 100 if len(records) > 0 else 100
            
            print(f"Token 151668 count: {count_151668}, Contains 151645: {contains_151645}, Fallback rate: {fallback_rate:.2f}%")
            
            # Early exit if token validation or fallback rate fails
            if count_151668 != 1 or not contains_151645 or fallback_rate > 50:
                print("❌ Failed token validation or fallback rate check")
                continue
            
            # Third check: Extract boxed answer
            boxed_answer = extract_boxed_answer(records, small_tokenizer)
                        
            # Early exit if boxed answer extraction fails
            if not boxed_answer:
                print("❌ Failed to extract boxed answer")
                continue
            
            print(f"Boxed answer: '{boxed_answer}', Expected: '{expected_answer}'")

            # Fourth check: Hard match
            hard_match = boxed_answer == expected_answer
            print(f"Hard match: {hard_match}")
            
            # If hard match passes, we have a good trace
            if hard_match:
                candidate_traces.append(records)
                print("✅ Trace stored in candidate pool (hard match)")
                continue
            
            # Fifth check: Soft match (only if hard match failed)
            llm_grade = llm_grader(expected_answer, boxed_answer, client)
            soft_match = llm_grade == 'true'
            print(f"Soft match: {soft_match}")
            
            # If soft match passes, we have a good trace
            if soft_match:
                candidate_traces.append(records)
                print("✅ Trace stored in candidate pool (soft match)")
            else:
                print("❌ Failed both hard and soft match")

        if candidate_traces:
            # Find shortest trace among candidates
            shortest_trace = min(candidate_traces, key=len)
            
            # U+FFFD FIX: Reconstruct text by decoding only the generated token IDs
            generated_ids = []
            generated_ids.extend([record['token_id'] for record in shortest_trace])
            
            # Decode the generated part
            generated_text = small_tokenizer.decode(generated_ids)
            
            # Safely update only this entry
            success = update_entry_trace(dataset_file, i, generated_text)
            
            if success:
                good_traces_count += 1
                print(f"Good trace stored in entry {i}")
            else:
                print(f"Failed to save trace for entry {i}")
        else:
            print(f"No generated trace matches the qualification. Leaving entry {i} empty")
        
        # Progress report
        questions_done = TARGET_INDICES.index(i) + 1
        total_questions = len(TARGET_INDICES)
        print(f"Progress: {questions_done}/{total_questions} questions done, {good_traces_count} entries with good traces so far")

if __name__ == "__main__":
    asyncio.run(main())