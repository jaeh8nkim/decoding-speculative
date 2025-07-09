"""
steacher_indices.py
------------------
Process specific indices using the downloaded model from steacher_download.py

Usage:
python steacher_indices.py <INDEX_RANGE> [GPU_INDEX]
python steacher_indices.py "111:121,131,134:173" 0
python steacher_indices.py "250:499" 1

  7B, fp16:  14 GB + kv cache 8192:   ~4 GB per sampling = ~22 GB (~0.50)
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm

# Path configuration (same as steacher_download.py)
USER = os.environ["USER"]
SCRATCH = "scratch2"  # change to "scratch" if needed

BASE_DIR = Path(f"/{SCRATCH}/{USER}")
MODEL_DIR = BASE_DIR / "models" / "s1.1-7B"
CACHE_DIR = BASE_DIR / "hf_cache"

# Parse command line arguments
if len(sys.argv) < 2 or len(sys.argv) > 3:
    print("Usage: python steacher_indices.py <INDEX_RANGE> [GPU_INDEX]")
    print("Examples:")
    print("  python steacher_indices.py \"111:121,131,134:173\"")
    print("  python steacher_indices.py \"250:499\" 0")
    sys.exit(1)

DF_INDEX_STRING = sys.argv[1]
LARGE_GPU_INDEX = sys.argv[2] if len(sys.argv) == 3 else "0"

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
    print(f"Using GPU: {LARGE_GPU_INDEX}")
    print(f"Model directory: {MODEL_DIR}")
except ValueError as e:
    print(f"Error parsing index string '{DF_INDEX_STRING}': {e}")
    sys.exit(1)

# Check if model directory exists
if not MODEL_DIR.exists():
    print(f"Error: Model directory {MODEL_DIR} not found!")
    print("Please run steacher_download.py first to download the model.")
    sys.exit(1)

# Simple Decoding with vLLM in Token ID Space
#
# - Large model generates tokens sequentially
# - Operates entirely in token ID space using vLLM's TokensPrompt to avoid Unicode issues (U+FFFD)
#   that commonly occur with rare mathematical symbols in reasoning traces when using BPE tokenization

# --------------------------- imports ---------------------------------------
import uuid, asyncio, contextlib, nest_asyncio, logging

import torch
from vllm import TokensPrompt
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams, RequestOutputKind

nest_asyncio.apply()
torch.set_grad_enabled(False)
logging.disable(logging.INFO)

# --------------------------- configuration ---------------------------------
LARGE_TEMPERATURE = 0.7
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
    global large_engine, large_tokenizer, large_vocab_size
    
    # Use local model directory instead of downloading
    large_checkpoint = str(MODEL_DIR)

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

    # Get model config using async method
    large_model_config = await large_engine.get_model_config()
    large_vocab_size = large_model_config.get_vocab_size()
    
    print(f"Large vocab size: {large_vocab_size}")

# --------------------------- sampling params -------------------------------
large_sampling_params = SamplingParams(
    max_tokens  = 1,
    temperature = LARGE_TEMPERATURE,
    top_p       = 0.95, 
    logprobs    = 20,
    output_kind = RequestOutputKind.DELTA,
)

# ------------------------- core decode loop --------------------------------
async def one_step(engine, sampling_params, context_ids):
    # Pass token IDs directly to vLLM using TokensPrompt
    # This avoids any decoding issues with partial tokens
    tokens_prompt = TokensPrompt(prompt_token_ids=context_ids)
    generator = engine.generate(tokens_prompt, sampling_params, request_id=str(uuid.uuid4()))
    return (await anext(generator)).outputs[0]

async def large_decode(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS):
    # Tokenize the prompt to IDs using large tokenizer
    context_ids_large = large_tokenizer.encode(prompt)
    
    step_index = 0

    # Create tqdm progress bar
    pbar = tqdm(total=max_new_tokens, desc="Generating tokens", unit="tok")

    for _ in range(max_new_tokens):
        large_output = await one_step(large_engine, large_sampling_params, context_ids_large)

        # Extract probabilities from large model output - logprobs is a list
        large_logprobs_dict = large_output.logprobs[0]  
        large_probs = {}
        for token_id, logprob in large_logprobs_dict.items():
            large_probs[token_id] = torch.exp(torch.tensor(logprob.logprob)).item()
        
        # Sample from large model's distribution
        idx_pool = torch.tensor(list(large_probs.keys()))
        prob_pool = torch.tensor(list(large_probs.values()))
        prob_pool = prob_pool / prob_pool.sum()  # Normalize for sampling
        
        pool_idx = torch.multinomial(prob_pool, 1).item()
        chosen_id = idx_pool[pool_idx].item()

        # Get token text for display
        chosen_text = large_tokenizer.decode([chosen_id])
        large_probability = large_probs.get(chosen_id, 0.0)

        step_index += 1
        record = dict(
            idx=step_index, 
            text=chosen_text,  # For display
            token_id=chosen_id,
            large_probability=large_probability,
        )
        yield record

        # Update progress bar
        pbar.update(1)

        # Append to context in ID space
        context_ids_large.append(chosen_id)
        
        if chosen_id == large_tokenizer.eos_token_id:
            break
    
    pbar.close()

# ---------------------- high-level convenience -----------------------------
async def run_large_decode(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS):
    records = []
    
    async for record in large_decode(prompt, max_new_tokens):
        records.append(record)
    
    return records

# --------------------------- database functions ----------------------------
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

def find_sequence_positions(token_ids, sequence):
    """Find all positions where a sequence of tokens occurs"""
    positions = []
    seq_len = len(sequence)
    
    for i in range(len(token_ids) - seq_len + 1):
        if token_ids[i:i + seq_len] == sequence:
            positions.append(i)
    
    return positions

def extract_boxed_answer(records, tokenizer):
    """Extract the last \\boxed{} answer between token sequence [151644, 9217] and token 151645"""
    token_ids = [record['token_id'] for record in records]
    
    # Find positions of the token sequence [151644, 9217]
    start_sequence = [151644, 9217]
    pos_start_sequence = find_sequence_positions(token_ids, start_sequence)
    pos_151645 = [i for i, tid in enumerate(token_ids) if tid == 151645]
    
    if len(pos_start_sequence) != 1 or len(pos_151645) == 0:
        return None

    start_pos = pos_start_sequence[0]
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
    
    # dataset_file = 'dataset_4qwen3.db'
    dataset_file = 'dataset_s1p17b.db'
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
            f"<|im_start|>assistant\n<|im_start|>think"
        )

        candidate_traces = []
        
        for j in range(NUM_SAMPLINGS):
            print(f"Started working on entry {i}, trial {j}")

            records = await run_large_decode(input)
            
            # First check: Token count
            print(f"Token count: {len(records)}")
            
            # Second check: Token validation
            token_ids = [record['token_id'] for record in records]
            start_sequence = [151644, 9217]
            pos_start_sequence = find_sequence_positions(token_ids, start_sequence)
            contains_151645 = 151645 in token_ids
            
            print(f"Token sequence [151644, 9217] count: {len(pos_start_sequence)}, Contains 151645: {contains_151645}")
            
            # Early exit if token validation fails
            if len(pos_start_sequence) != 1 or not contains_151645:
                print("❌ Failed token validation")
                continue
            
            # Third check: Extract boxed answer
            boxed_answer = extract_boxed_answer(records, large_tokenizer)
                        
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
            
            # Reconstruct text by decoding only the generated token IDs
            generated_ids = []
            generated_ids.extend([record['token_id'] for record in shortest_trace])
            
            # Decode the generated part
            generated_text = large_tokenizer.decode(generated_ids)
            
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