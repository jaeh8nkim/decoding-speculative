"""
Example usage:
python eval.py 0 1 16384 "aime_2024,aime_2025,gpqa_diamond,math_500"
python eval.py 3 16 8192 "aime_2025"
"""

import sys
import asyncio

if len(sys.argv) != 5:
    print("Usage: python eval.py <SMALL_GPU_INDEX> <NUM_RUNS> <MAX_SEQ_LEN> <EVAL_DATASETS>")
    sys.exit(1)

SMALL_GPU_INDEX = sys.argv[1]
NUM_RUNS    = int(sys.argv[2])
MAX_SEQ_LEN = int(sys.argv[3])
EVAL_DATASETS   = sys.argv[4]

# Imports and Configuration
import os
import json
import torch
import re
from collections import Counter
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from pathlib import Path
import random, os, json, re, torch
import openai
from dotenv import load_dotenv

# vLLM imports
import html, uuid, asyncio, contextlib, nest_asyncio, logging
from IPython.display import HTML, display
from huggingface_hub import snapshot_download
from vllm import TokensPrompt
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams, RequestOutputKind

load_dotenv()
nest_asyncio.apply()
torch.set_grad_enabled(False)
logging.disable(logging.INFO)

BASE_SEED = 42
SMALL_TEMPERATURE = 0.7

# MODEL_NAME = "Qwen3-0.6B"
# MODEL_REMOTE_PATH = "Qwen/" + MODEL_NAME
# MODEL_LOCAL_PATH = "vanilla-qwen3-0.6b-local"

# MODEL_NAME = "s1-slth-qwen3-0.6b"
# MODEL_REMOTE_PATH = "jaeh8nkim/" + MODEL_NAME
# MODEL_LOCAL_PATH = "s1-slth-qwen3-0.6b-local"

MODEL_NAME = "s1-4q36-qwen3-0.6b"
MODEL_REMOTE_PATH = "jaeh8nkim/" + MODEL_NAME
MODEL_LOCAL_PATH = "s1-4q36-qwen3-0.6b-local"

# Global variables for engine and tokenizer
engine = None
tokenizer = None
vocab_size = None


def download_model_locally(repo_name, local_path):
    """Download model from HuggingFace and save locally"""
    print(f"📥 Downloading model from {repo_name}...")
    
    try:
        # Check if model already exists locally
        if os.path.exists(local_path):
            print(f"✅ Model already exists at {local_path}")
            return local_path
        
        # Download model using snapshot_download (same as vLLM uses)
        checkpoint_path = snapshot_download(repo_name)
        
        # Create local directory
        os.makedirs(local_path, exist_ok=True)
        
        # Copy all files from checkpoint to local path
        import shutil
        shutil.copytree(checkpoint_path, local_path, dirs_exist_ok=True)
        
        print(f"✅ Model downloaded and saved to {local_path}")
        return local_path
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None


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
async def setup_engine():
    global engine, tokenizer, vocab_size
    
    # Use the locally downloaded model
    print(f"Setting up engine with local model: {MODEL_LOCAL_PATH}")

    with visible_gpus(SMALL_GPU_INDEX):
        print("torch sees", torch.cuda.device_count(), "GPU(s)")              
        engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(model=MODEL_LOCAL_PATH,  # Use local path instead of checkpoint
                            tensor_parallel_size=1,
                            max_model_len=MAX_SEQ_LEN, 
                            gpu_memory_utilization=0.90,
                            dtype="bfloat16"),
            start_engine_loop=True)
        
        tokenizer = await engine.get_tokenizer()

    # Get model config using async method
    model_config = await engine.get_model_config()
    vocab_size = model_config.get_vocab_size()
    
    print(f"Vocab size: {vocab_size}")


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


# Evaluation functions
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

async def graded_is_correct(gold, pred, tokenizer):
    # Convert generated text into token-records so extract_boxed_answer works
    ids = tokenizer.encode(pred)
    records = [{"token_id": t} for t in ids]

    boxed = extract_boxed_answer(records, tokenizer)
    extracted = boxed if boxed else pred

    return llm_grader(gold, extracted, client) == "true"

def print_dataset_info(dataset, task_name):
    """Print dataset count and first 5 examples"""
    print(f"\n--- {task_name.upper()} DATASET INFO ---")
    print(f"Total samples: {len(dataset)}")
    print(f"Dataset type: {type(dataset)}")
    
    # Check the first item to understand the structure
    if len(dataset) > 0:
        first_item = dataset[0]
        print(f"First item type: {type(first_item)}")
        print(f"First item keys: {list(first_item.keys()) if hasattr(first_item, 'keys') else 'No keys'}")
    
    print(f"First 5 examples:")
    
    for i in range(min(5, len(dataset))):
        item = dataset[i]  # Access by index instead of iteration
        print(f"\n{i+1}. ", end="")
        
        try:
            if "math" in task_name:
                question = item.get("problem", item.get("question", ""))
                answer = item.get("solution", "")
                print(f"Question: {question[:200]}...")
                print(f"   Answer: {answer[:200]}...")
            elif "gpqa" in task_name:
                # Handle both original GPQA format and alternative formats
                question = item.get("Question", item.get("question", item.get("Problem", "")))
                answer = item.get("Correct Answer", item.get("correct_answer", item.get("Answer", "")))
                print(f"Question: {question[:200]}...")
                print(f"   Answer: {answer}")
            else:  # AIME
                question = item.get("problem", item.get("Problem", item.get("question", "")))
                answer = str(item.get("answer", item.get("Answer", "")))
                print(f"Question: {question[:200]}...")
                print(f"   Answer: {answer}")
        except Exception as e:
            print(f"Error displaying item: {e}")
            print(f"Item keys: {list(item.keys()) if hasattr(item, 'keys') else 'Not a dict'}")
            print(f"Item type: {type(item)}")
            print(f"Raw item: {str(item)[:200]}...")
            break

async def evaluate_problem_multiple_times(item, task_name, num_runs, sampling_params):
    """
    Evaluate a single problem multiple times and return accuracy for that problem.
    """
    global engine, tokenizer
    correct = 0
    
    # Extract question and gold answer based on task type
    try:
        if "math" in task_name:
            question = item.get("problem", item.get("question", ""))
            gold = item.get("solution", "")
        elif "gpqa" in task_name:
            # Handle both original GPQA format and alternative formats
            question = item.get("Question", item.get("question", item.get("Problem", "")))
            
            # Try different field names for correct answer
            gold = item.get("Correct Answer", item.get("correct_answer", item.get("Answer", "")))
            
            # Handle choices if they exist
            choices = []
            if "Incorrect Answer 1" in item:
                # Original format
                choices = [
                    item["Incorrect Answer 1"],
                    item["Incorrect Answer 2"],
                    item["Incorrect Answer 3"],
                    item["Correct Answer"],
                ]
                random.shuffle(choices)
                gold = chr(65 + choices.index(item["Correct Answer"]))
                question += "\n\nChoices:\n" + "\n".join(
                    f"{chr(65+i)}. {c}" for i, c in enumerate(choices)
                )
            elif any(f"choice_{i}" in item for i in ['A', 'B', 'C', 'D']):
                # Alternative choice format
                choices = [item.get(f"choice_{i}", "") for i in ['A', 'B', 'C', 'D']]
                question += "\n\nChoices:\n" + "\n".join(
                    f"{chr(65+i)}. {c}" for i, c in enumerate(choices) if c
                )
                # Find which choice is correct
                for i, choice in enumerate(['A', 'B', 'C', 'D']):
                    if item.get(f"choice_{choice}", "") == gold:
                        gold = choice
                        break
            
        else:  # AIME
            question = item.get("problem", item.get("Problem", item.get("question", "")))
            gold = str(item.get("answer", item.get("Answer", "")))

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
        
        # Run the problem multiple times
        for run in range(num_runs):
            random.seed(BASE_SEED + run)
            torch.manual_seed(BASE_SEED + run)
            
            # Generate with vLLM (using working pattern)
            request_id = str(uuid.uuid4())
            generator = engine.generate(input, sampling_params, request_id)
            
            # Get the result using working pattern
            final_output = None
            async for request_output in generator:
                final_output = request_output
            
            if final_output and final_output.outputs:
                predicted = final_output.outputs[0].text.strip()
                if await graded_is_correct(gold, predicted, tokenizer):
                    correct += 1
                
    except Exception as e:
        print(f"Error processing problem: {e}")
        return 0
    
    return correct / num_runs

def load_combined_aime_2025():
    """Load and combine both AIME2025-I and AIME2025-II datasets"""
    try:
        # Load both AIME2025 datasets silently
        aime_i = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test", trust_remote_code=True)
        aime_ii = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test", trust_remote_code=True)
        
        # Combine the datasets
        from datasets import concatenate_datasets
        combined_aime = concatenate_datasets([aime_i, aime_ii])
        
        return combined_aime
        
    except Exception as e:
        print(f"❌ Error loading AIME2025 datasets: {e}")
        return None

async def evaluate_model_average(num_runs, eval_datasets):
    """
    Evaluate a model on multiple datasets with multiple runs.
    """
    
    # Parse evaluation datasets from config string
    evaluation_order = [dataset.strip() for dataset in eval_datasets.split(",")]
    print(f"📋 Will evaluate datasets: {', '.join(evaluation_order)}")

    # First, load all datasets and print their info
    datasets = {}
    print("\n" + "="*60)
    print(" LOADING ALL DATASETS ")
    print("="*60)

    for dataset_name in evaluation_order:
        if dataset_name == "aime_2024":
            try:
                print(f"Loading aime_2024 dataset...")
                ds = load_dataset("HuggingFaceH4/aime_2024", split="train", trust_remote_code=True)
                datasets["aime_2024"] = ds
                print(f"✅ aime_2024 loaded: {len(ds)} problems")
            except Exception as e:
                print(f"❌ Error loading aime_2024: {e}")
        
        elif dataset_name == "aime_2025":
            try:
                print(f"Loading aime_2025 dataset...")
                aime_2025_combined = load_combined_aime_2025()
                if aime_2025_combined is not None:
                    datasets["aime_2025"] = aime_2025_combined
                    print(f"✅ aime_2025 loaded: {len(aime_2025_combined)} problems")
            except Exception as e:
                print(f"❌ Error loading aime_2025: {e}")
        
        elif dataset_name == "gpqa_diamond":
            try:
                print(f"Loading gpqa_diamond dataset...")
                ds = load_dataset("spawn99/GPQA-diamond-ClaudeR1", split="train", trust_remote_code=True)
                datasets["gpqa_diamond"] = ds
                print(f"✅ gpqa_diamond loaded: {len(ds)} problems")
            except Exception as e:
                print(f"❌ Error loading gpqa_diamond: {e}")
        
        elif dataset_name == "math_500":
            try:
                print(f"Loading math_500 dataset...")
                ds = load_dataset("HuggingFaceH4/MATH-500", split="test", trust_remote_code=True)
                datasets["math_500"] = ds
                print(f"✅ math_500 loaded: {len(ds)} problems")
            except Exception as e:
                print(f"❌ Error loading math_500: {e}")
        
        else:
            print(f"⚠️  Unknown dataset: {dataset_name}")
    
    # Print info for all loaded datasets in the desired order
    print("\n" + "="*60)
    print(" DATASET INFORMATION ")
    print("="*60)
    
    for task in evaluation_order:
        if task in datasets:
            print_dataset_info(datasets[task], task)
    
    # Now run evaluations in the specified order
    print("\n" + "="*60)
    print(" STARTING EVALUATIONS ")
    print("="*60)
    
    all_results = {}
    
    # --------------------------- sampling params -------------------------------
    sampling_params = SamplingParams(
        max_tokens=MAX_SEQ_LEN,
        temperature=SMALL_TEMPERATURE,
        top_p=0.95,
    )
    
    for task in evaluation_order:
        if task not in datasets:
            print(f"⚠️  Skipping {task} - dataset not loaded")
            continue
            
        ds = datasets[task]
        print(f"\n🔄 Evaluating {task.upper()}...")
        
        problem_accuracies = []
        
        # Iterate through each problem
        for i in tqdm(range(len(ds)), desc=f"{task} problems"):
            item = ds[i]
            problem_accuracy = await evaluate_problem_multiple_times(item, task, num_runs, sampling_params)
            problem_accuracies.append(problem_accuracy)
            
            # Determine status
            correct_runs = int(problem_accuracy * num_runs)  # Convert back to count
            if correct_runs > 0:
                status_emoji = "✅"
            else:
                status_emoji = "❌"
            
            # Show individual problem results with additional info for AIME 2025
            if task == "aime_2025":
                # Determine if this is from AIME I or II based on position
                if i < 15:  # First 15 are from AIME I
                    contest_info = f"(AIME I, #{i+1})"
                else:  # Last 15 are from AIME II
                    contest_info = f"(AIME II, #{i-14})"
                print(f"{status_emoji} Problem {i+1:02d}/{len(ds)} {contest_info} — {task}: {correct_runs}/{num_runs}")
            else:
                print(f"{status_emoji} Problem {i+1:02d}/{len(ds)} — {task}: {correct_runs}/{num_runs}")

        average_accuracy = sum(problem_accuracies) / len(problem_accuracies)
        all_results[task] = {"average_accuracy": average_accuracy, "problem_accuracies": problem_accuracies}
        
        # Final summary for this dataset
        total_runs = len(ds) * num_runs
        total_correct_runs = sum(int(acc * num_runs) for acc in problem_accuracies)
        print(f"✅ {task} complete - {total_correct_runs}/{total_runs} ({average_accuracy:.2%} accuracy)")

    return all_results


async def main():
    """Main function that runs the evaluation pipeline"""
    print(f"🚀 Starting evaluation with:")
    print(f"   SMALL_GPU_INDEX: {SMALL_GPU_INDEX}")
    print(f"   NUM_RUNS: {NUM_RUNS}")
    print(f"   MAX_SEQ_LEN: {MAX_SEQ_LEN}")
    print(f"   EVAL_DATASETS: {EVAL_DATASETS}")
    
    # Download the model locally
    print("🚀 Downloading model locally...")
    model_path = download_model_locally(MODEL_REMOTE_PATH, MODEL_LOCAL_PATH)

    if model_path is None:
        raise RuntimeError("Failed to download model")

    # Initialize the engine
    await setup_engine()
    
    # Run evaluation
    results = await evaluate_model_average(NUM_RUNS, EVAL_DATASETS)

    print("\n" + "="*50)
    print(" FINAL AVERAGED ACCURACIES ")
    print("="*50)
    for task, result in results.items():
        print(f"{task.upper():<15}: {result['average_accuracy']:.2%}")
    print("="*50)


if __name__ == "__main__":
    asyncio.run(main())