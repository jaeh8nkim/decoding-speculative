"""
python datagen.py 2 0 0 249
python datagen.py 3 1 250 499

python datagen.py 2 0 500 749
python datagen.py 3 1 750 999

  7B, fp16:  14 GB + kv cache 8192:   ~4 GB per sampling = ~22 GB (~0.50)
0.6B, fp16: 1.2 GB + kv cache 8192: ~0.8 GB per sampling =  ~3 GB (~0.10)
"""

import sys
from tqdm import tqdm

if len(sys.argv) != 5:
    print("Usage: python datagen.py <LARGE_GPU_INDEX> <SMALL_GPU_INDEX> <DF_START_INDEX> <DF_END_INDEX>")
    sys.exit(1)

LARGE_GPU_INDEX = sys.argv[1]
SMALL_GPU_INDEX = sys.argv[2]
DF_START_INDEX  = int(sys.argv[3])
DF_END_INDEX    = int(sys.argv[4])

# Reverse Speculative Decoding with vLLM

# large model proposes, small model filters
# accept if large model token is within small model top 20 and has prob over 0.01
# vocab match mask also
# multiple trials on large model proposals
# live per-token stat
# html heatmap
# closely resembling HF version

# --------------------------- imports ---------------------------------------
import os, html, uuid, asyncio, contextlib, nest_asyncio, logging
from IPython.display import HTML, display

import torch
from huggingface_hub import snapshot_download
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
    global large_vocab_size, small_vocab_size, vocab_match_mask
    
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
                            dtype="float16"),
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

# -------------------------- helper functions -------------------------------
async def one_step(engine, sampling_params, context):
    generator = engine.generate(context, sampling_params, request_id=str(uuid.uuid4()))
    return (await anext(generator)).outputs[0]

def html_heat(records):
    probability_min, probability_max = 0.0, 0.2
    def colour(probability):
        if probability >= probability_max: 
            return "rgb(0,0,0)"
        red = int(255 * (probability_max - probability) / (probability_max - probability_min))
        return f"rgb({red},0,0)"
    spans = []
    for record in records:
        text = html.escape(record['text']).replace(" ", "&nbsp;")
        style = f"color:{colour(record['small_probability'])};"
        if record['fallback']: 
            style += " text-decoration:underline;"
        spans.append(f"<span style='{style}'>{text}</span>")
    return HTML("<pre style='white-space:pre-wrap; line-height:1.45; "
                "font-family:inherit; background:#fff; padding:8px; "
                "border:1px solid #ddd;'>" + "".join(spans) + "</pre>")

# ------------------------- core decode loop --------------------------------
async def mixed_decode(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS):
    context = prompt
    step_index = 0
    PROB_THRESHOLD = 0.01
    NUM_TRIALS = 5 

    # Create tqdm progress bar
    pbar = tqdm(total=max_new_tokens, desc="Generating tokens", unit="tok")

    for _ in range(max_new_tokens):
        large_output, small_output = await asyncio.gather(
            one_step(large_engine, large_sampling_params, context),
            one_step(small_engine, small_sampling_params, context))

        # if step_index < 3:
        #     print(f"  large_output: {large_output}")
        #     print(f"  small_output: {small_output}")

        # Extract probabilities from large model output - logprobs is a list
        large_logprobs_dict = large_output.logprobs[0]  
        large_probs = {}
        for token_id, logprob in large_logprobs_dict.items():
            if vocab_match_mask[token_id] > 0:  # Only include vocab-matched tokens
                large_probs[token_id] = torch.exp(torch.tensor(logprob.logprob)).item()  # Access .logprob attribute
        
        idx_pool = torch.tensor(list(large_probs.keys()))
        prob_pool = torch.tensor(list(large_probs.values()))
        prob_pool = prob_pool / prob_pool.sum()  # Normalize for sampling

        # Extract probabilities from small model output - compact dict for lookup  
        small_logprobs_dict = small_output.logprobs[0] 
        small_probs = {}
        for token_id, logprob in small_logprobs_dict.items():
            small_probs[token_id] = torch.exp(torch.tensor(logprob.logprob)).item()  # Access .logprob attribute

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

        # Get token text and probabilities for the chosen token
        chosen = small_tokenizer.decode([chosen_id])
        large_probability = large_probs.get(chosen_id, 0.0)
        small_probability = small_probs.get(chosen_id, 0.0)

        step_index += 1
        record = dict(
            idx=step_index, 
            text=chosen, 
            token_id=chosen_id,
            fallback=fallback, 
            large_probability=large_probability, 
            small_probability=small_probability,
        )
        yield record

        # print(f"{step_index:4d}{'*' if fallback else ' '}\t"
        #       f"{large_probability:.4f}\t{small_probability:.4f}\t"
        #       f"{chosen_id}\t'{chosen}'",
        #       flush=True)

        # Update progress bar
        pbar.update(1)

        context += chosen
        if chosen_id == small_tokenizer.eos_token_id:
            break

# ---------------------- high-level convenience -----------------------------
async def run_mixed_decode(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS):
    # print("-"*80)
    # print("Step\tL_Prob\tS_Prob\tTok_ID\tTok_Txt")
    records = []
    async for record in mixed_decode(prompt, max_new_tokens):
        records.append(record)
    # print("-"*80)
    # display(html_heat(records))
    # fallback_count = sum(record['fallback'] for record in records)
    # print(f"Fallback tokens: {fallback_count}/{len(records)} "
    #       f"({fallback_count/len(records)*100:.2f} %)")
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

def extract_boxed_answer(records):
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

    # Extract text between the tokens
    between_text = ''.join(record['text'] for record in records[start_pos:end_pos+1])
    
    # Find all \\boxed{} patterns
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(boxed_pattern, between_text)
    
    if matches:
        return matches[-1]  # Return the last match
    return None

def llm_grader(expected_answer, boxed_answer, openai_client, model_name="gpt-4o-mini"):

    def grader_prompt(expected_answer, boxed_answer):
        """Creates the system and user prompts for grading."""
        system_prompt = (
            "You are an expert grader tasked with evaluating the correctness of an answer.\n"
            "You will be provided with two pieces of text: the expected answer and the generated answer.\n"
            "Your task is to determine if the generated answer is semantically equivalent to the expected answer.\n"
            "Ignore minor formatting differences, extra whitespace, or trivial variations. For numerical answers, consider equivalent representations as correct (e.g., '1/2' and '0.5').\n"
            "Respond with exactly one word: either 'true' (if correct) or 'false' (if incorrect). Do not include quotation marks, explanations, or any other text.\n"
        )
        user_prompt = (
            "Expected answer:\n"
            f"{expected_answer}\n"
            "Generated answer:\n"
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
    # Fire up the engines
    await setup_engines()
    
    dataset_file = 'dataset_4qwen3.db'
    good_traces_count = 0

    for i in range(DF_START_INDEX, DF_END_INDEX + 1):
        # Check if entry already has a trace (thread-safe)
        existing_trace = read_entry_trace(dataset_file, i)
        
        if existing_trace:
            good_traces_count += 1
            print(f"Entry {i} already has a trace, skipping")
            continue
            
        # Get question and answer for this entry
        question, expected_answer = read_entry_qanda(dataset_file, i)

        prompt = f"""A conversation between User and Assistant. The User asks a question, and the Assistant responds in two clearly defined sections: 1. Reasoning Process - A step-by-step, logical exploration and analysis of the problem, enclosed within <think> and </think> tags. 2. Answer - A direct and concise response based on the reasoning process, with the final answer enclosed within \\boxed{{}}. For example, 
<think>
reasoning process here
</think>
answer here
\\boxed{{final answer here}}

Now, continue the actual conversation below.
User: {question}
Assistant:
<think>"""
        
        candidate_traces = []
        
        for j in range(16):
            print(f"Started working on entry {i}, trial {j}")

            records = await run_mixed_decode(prompt)
            
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
            boxed_answer = extract_boxed_answer(records)
                        
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
            
            # Reconstruct the generated text from the trace
            generated_text = ''.join(record['text'] for record in shortest_trace)
            
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
        questions_done = i + 1 - DF_START_INDEX
        total_questions = DF_END_INDEX + 1 - DF_START_INDEX
        print(f"Progress: {questions_done} questions done, {good_traces_count} entries with good traces so far, {total_questions} total questions")

if __name__ == "__main__":
    asyncio.run(main())