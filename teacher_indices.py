"""
python teacher_indices.py 2 111:121,131,134:173
python teacher_indices.py 3 250:499

  7B, fp16:  14 GB + kv cache 8192:   ~4 GB per sampling = ~22 GB (~0.50)
"""

import sys
from tqdm import tqdm

if len(sys.argv) != 3:
    print("Usage: python teacher_indices.py <LARGE_GPU_INDEX> <DF_INDEX_STRING>")
    sys.exit(1)

LARGE_GPU_INDEX = sys.argv[1]
DF_INDEX_STRING = sys.argv[2]

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

# Simple Decoding with vLLM using LLM.generate
#
# - Large model generates tokens using batch generation
# - Maintains same interface as async version for compatibility

# --------------------------- imports ---------------------------------------
import os, contextlib, logging, time, sys
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

torch.set_grad_enabled(False)

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

# --------------------------- configuration ---------------------------------
LARGE_MODEL_NAME  = "simplescaling/s1.1-7B"

LARGE_TEMPERATURE = 0.7
MAX_SEQ_LEN       = 8192
MAX_NEW_TOKENS    = MAX_SEQ_LEN - 1024

# --------------------------- model setup -----------------------------------
def setup_large_model():
    global large_model, large_tokenizer, large_sampling_params
    
    with visible_gpus(LARGE_GPU_INDEX):
        print(f"Setting up model on GPU {LARGE_GPU_INDEX}")
        print("torch sees", torch.cuda.device_count(), "GPU(s)")
        
        large_model = LLM(
            LARGE_MODEL_NAME,
            tensor_parallel_size=1,
            dtype="float16",
            max_model_len=MAX_SEQ_LEN,
            gpu_memory_utilization=0.45,
            # Remove device parameter - let CUDA_VISIBLE_DEVICES handle it
        )
        
        large_tokenizer = AutoTokenizer.from_pretrained(LARGE_MODEL_NAME)
        
        # Get stop token IDs
        stop_token_ids = large_tokenizer("<|im_end|>")["input_ids"]
        
        large_sampling_params = SamplingParams(
            max_tokens=MAX_NEW_TOKENS,
            min_tokens=0,
            temperature=LARGE_TEMPERATURE,
            top_p=0.95,
            stop_token_ids=stop_token_ids,
        )
        
        print(f"Large vocab size: {large_tokenizer.vocab_size}")

# ------------------------- core decode loop --------------------------------
def large_generate(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS):
    # Generate full output using LLM.generate
    start_time = time.time()
    
    outputs = large_model.generate(prompt, large_sampling_params)
    
    generated_text = outputs[0].outputs[0].text
    termination_type = outputs[0].outputs[0].finish_reason
    
    end_time = time.time()
    generation_time = end_time - start_time
    minutes, seconds = divmod(generation_time, 60)
    print(f"Generation completed in {int(minutes):02d}:{int(seconds):02d} ({termination_type})")
    
    print(prompt + generated_text)
    
    return generated_text, termination_type

# No wrapper needed - use large_generate directly

# ------------------------ fire up the model --------------------------------
setup_large_model()

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

def extract_boxed_answer(generated_text):
    """Extract the last \\boxed{} answer from the entire generated text"""
    
    # Search the entire generated text for \boxed{} patterns
    matches = []
    i = 0
    while i < len(generated_text):
        boxed_start = generated_text.find('\\boxed{', i)
        if boxed_start == -1:
            break
        
        j = boxed_start + 7  # Start after '\boxed{'
        brace_count = 1
        while j < len(generated_text) and brace_count > 0:
            if generated_text[j] == '{':
                brace_count += 1
            elif generated_text[j] == '}':
                brace_count -= 1
            j += 1
        
        if brace_count == 0:
            matches.append(generated_text[boxed_start + 7:j-1])
        
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

def main():
    NUM_SAMPLINGS = 16

    # Fire up the model (already done globally)
    # setup_large_model()  # This was already called
    
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

        found_good_trace = False
        
        for j in range(NUM_SAMPLINGS):
            print(f"Started working on entry {i}, trial {j}")

            generated_text, termination_type = large_generate(input)
            
            # First check: Text length
            token_count = len(large_tokenizer.encode(generated_text))
            print(f"Generated text length: {token_count} tokens")
            
            # Second check: Termination validation - should stop due to stop token, not max length
            stopped_properly = termination_type == "stop"
            has_answer_start = '<|im_start|>answer' in generated_text
            
            print(f"Stopped properly: {stopped_properly}, Has '<|im_start|>answer': {has_answer_start}")
            
            # Early exit if validation fails
            if not stopped_properly:
                print("❌ Failed text validation")
                continue
            
            # Third check: Extract boxed answer
            boxed_answer = extract_boxed_answer(generated_text)
            
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
                success = update_entry_trace(dataset_file, i, generated_text)
                if success:
                    good_traces_count += 1
                    print("✅ Good trace stored (hard match)")
                    found_good_trace = True
                    break
                else:
                    print(f"Failed to save trace for entry {i}")
                continue
            
            # Fifth check: Soft match (only if hard match failed)
            llm_grade = llm_grader(expected_answer, boxed_answer, client)
            soft_match = llm_grade == 'true'
            print(f"Soft match: {soft_match}")
            
            # If soft match passes, we have a good trace
            if soft_match:
                success = update_entry_trace(dataset_file, i, generated_text)
                if success:
                    good_traces_count += 1
                    print("✅ Good trace stored (soft match)")
                    found_good_trace = True
                    break
                else:
                    print(f"Failed to save trace for entry {i}")
            else:
                print("❌ Failed both hard and soft match")

        if not found_good_trace:
            print(f"No generated trace matches the qualification. Leaving entry {i} empty")
        
        # Progress report
        questions_done = TARGET_INDICES.index(i) + 1
        total_questions = len(TARGET_INDICES)
        print(f"Progress: {questions_done}/{total_questions} questions done, {good_traces_count} entries with good traces so far")

if __name__ == "__main__":
    main()