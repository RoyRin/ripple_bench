#!/usr/bin/env python3
"""
Evaluate a single model on Ripple Bench Dataset and save results to CSV

Usage:
    python evaluate_model_on_ripple.py <dataset_path> <model_name> --output-csv <output.csv>
    python evaluate_model_on_ripple.py --hf-dataset royrin/ripple-bench <model_name> --output-csv <output.csv>
    
Example:
    # Load from local file
    python evaluate_model_on_ripple.py data/ripple_bench.json HuggingFaceH4/zephyr-7b-beta --output-csv zephyr_base_results.csv
    
    # Load from Hugging Face
    python evaluate_model_on_ripple.py --hf-dataset royrin/ripple-bench HuggingFaceH4/zephyr-7b-beta --output-csv zephyr_base_results.csv
"""

import json
import argparse
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import gc
import shutil
import os

from ripple_bench.metrics import answer_single_question
from ripple_bench.utils import read_dict

# Set default HuggingFace cache directory via environment variables
DEFAULT_CACHE_DIR = "/n/home04/rrinberg/data_dir/HF_cache"
os.environ['HF_HOME'] = DEFAULT_CACHE_DIR
os.environ['HUGGINGFACE_HUB_CACHE'] = DEFAULT_CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = DEFAULT_CACHE_DIR

# Available models - use model name or HF path directly
MODELS = {
    # Base models
    "Llama-3-8b-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "zephyr-7b-beta": "HuggingFaceH4/zephyr-7b-beta",

    # LLM-GAT models
    "llama-3-8b-instruct-graddiff":
    "LLM-GAT/llama-3-8b-instruct-graddiff-checkpoint-8",
    "llama-3-8b-instruct-elm": "LLM-GAT/llama-3-8b-instruct-elm-checkpoint-8",
    "llama-3-8b-instruct-pbj": "LLM-GAT/llama-3-8b-instruct-pbj-checkpoint-8",
    "llama-3-8b-instruct-tar": "LLM-GAT/llama-3-8b-instruct-tar-checkpoint-8",
    "llama-3-8b-instruct-rmu": "LLM-GAT/llama-3-8b-instruct-rmu-checkpoint-8",
    "llama-3-8b-instruct-rmu-lat":
    "LLM-GAT/llama-3-8b-instruct-rmu-lat-checkpoint-8",
    "llama-3-8b-instruct-repnoise":
    "LLM-GAT/llama-3-8b-instruct-repnoise-checkpoint-8",
    "llama-3-8b-instruct-rr": "LLM-GAT/llama-3-8b-instruct-rr-checkpoint-8",

    # Zephyr unlearned models
    "zephyr-7b-elm": "baulab/elm-zephyr-7b-beta",
    "zephyr-7b-rmu": "cais/Zephyr_RMU",
}

# LLM-GAT models that have multiple checkpoints (base paths without checkpoint number)
LLM_GAT_MODELS = {
    "llama-3-8b-instruct-graddiff":
    "LLM-GAT/llama-3-8b-instruct-graddiff-checkpoint",
    "llama-3-8b-instruct-elm": "LLM-GAT/llama-3-8b-instruct-elm-checkpoint",
    "llama-3-8b-instruct-pbj": "LLM-GAT/llama-3-8b-instruct-pbj-checkpoint",
    "llama-3-8b-instruct-tar": "LLM-GAT/llama-3-8b-instruct-tar-checkpoint",
    "llama-3-8b-instruct-rmu": "LLM-GAT/llama-3-8b-instruct-rmu-checkpoint",
    "llama-3-8b-instruct-rmu-lat":
    "LLM-GAT/llama-3-8b-instruct-rmu-lat-checkpoint",
    "llama-3-8b-instruct-repnoise":
    "LLM-GAT/llama-3-8b-instruct-repnoise-checkpoint",
    "llama-3-8b-instruct-rr": "LLM-GAT/llama-3-8b-instruct-rr-checkpoint",
}

# Base models to keep in cache
BASE_MODELS = {"Llama-3-8b-Instruct", "zephyr-7b-beta"}


def load_dataset_from_hf(dataset_name: str):
    """Load Ripple Bench dataset from Hugging Face."""
    print(f"Loading dataset from Hugging Face: {dataset_name}")

    # Load the dataset
    dataset = load_dataset(dataset_name)

    # The dataset should have a 'train' split with the questions
    if 'train' in dataset:
        data = dataset['train']
    else:
        # If no train split, use the first available split
        data = dataset[list(dataset.keys())[0]]

    # Convert to the expected format
    questions = []
    for item in data:
        # Each item should have the question structure
        question = {
            'question': item.get('question', ''),
            'choices': item.get('choices', []),
            'answer': item.get('answer', ''),
            'topic': item.get('topic', 'unknown'),
            'distance': item.get('distance', -1),
            'id': item.get('id', '')
        }
        questions.append(question)

    return {'questions': questions}


def delete_model_from_cache(model_id: str, cache_dir: str = None):
    """Delete a model from the HuggingFace cache to save space."""
    if cache_dir is None:
        # Use our default cache location
        cache_dir = DEFAULT_CACHE_DIR

    # Convert model_id to cache folder name format
    model_cache_name = f"models--{model_id.replace('/', '--')}"
    model_path = Path(cache_dir) / model_cache_name

    if model_path.exists():
        print(f"Deleting cached model: {model_path}")
        shutil.rmtree(model_path)
        print(f"Deleted {model_id} from cache")
    else:
        print(f"Model {model_id} not found in cache at {model_path}")


def load_model(model_id: str, cache_dir: str = None):
    """Load a model from HuggingFace Hub."""
    print(f"Loading model: {model_id}")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32 if device == 'cpu' else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        cache_dir=cache_dir,
        device_map="auto" if device == 'cuda:0' else None)

    if device == 'cpu':
        model = model.to(device)

    model.requires_grad_(False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id,
                                              use_fast=False,
                                              cache_dir=cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

    return model, tokenizer


def evaluate_model(dataset_path: str,
                   model_name: str,
                   output_csv: str,
                   cache_dir: str = None,
                   hf_dataset: str = None,
                   delete_after: bool = False):
    """Evaluate a model on ripple bench dataset and save results to CSV.
    
    Args:
        dataset_path: Path to local dataset JSON file (mutually exclusive with hf_dataset)
        model_name: Name of the model to evaluate (can be a key from MODELS dict or HF path)
        output_csv: Path to save the results CSV
        cache_dir: Optional cache directory for models
        hf_dataset: Hugging Face dataset name (mutually exclusive with dataset_path)
        delete_after: Whether to delete the model from cache after evaluation
    """

    # Check if model_name is a key in MODELS dict
    if model_name in MODELS:
        model_id = MODELS[model_name]
        model_display_name = model_name
    else:
        # Assume it's a direct HF model path
        model_id = model_name
        model_display_name = model_name

    # Load dataset from either local file or Hugging Face
    if hf_dataset:
        print(f"Loading dataset from Hugging Face: {hf_dataset}")
        dataset = load_dataset_from_hf(hf_dataset)
    else:
        print(f"Loading dataset from {dataset_path}")
        dataset = read_dict(dataset_path)

    # Handle different dataset formats
    if 'questions' in dataset:
        # Direct questions list (old format)
        questions = dataset['questions']
    elif 'topics' in dataset:
        # Questions nested in topics (new format)
        questions = []
        for topic_data in dataset['topics']:
            if 'questions' in topic_data:
                # Add topic info to each question if not present
                for q in topic_data['questions']:
                    if 'topic' not in q:
                        q['topic'] = topic_data.get('topic', 'unknown')
                    if 'distance' not in q:
                        q['distance'] = topic_data.get('distance', -1)
                    questions.append(q)
    elif 'raw_data' in dataset and 'questions' in dataset['raw_data']:
        # Questions in raw_data section
        questions = dataset['raw_data']['questions']
    else:
        raise ValueError(
            "Cannot find questions in dataset. Expected 'questions', 'topics', or 'raw_data' key."
        )

    print(f"Loaded {len(questions)} questions")

    # Load model
    model, tokenizer = load_model(model_id, cache_dir)

    # Evaluate
    print(f"\nEvaluating {model_display_name}...")
    results = []

    for i, q in enumerate(tqdm(questions, desc="Evaluating")):
        try:
            # Validate question format
            if not isinstance(q.get('question'), str):
                print(
                    f"Warning: Question {i} has invalid question field: {type(q.get('question'))}"
                )
                print(f"Question data: {q}")
                continue

            if not isinstance(q.get('choices'), list):
                print(
                    f"Warning: Question {i} has invalid choices field: {type(q.get('choices'))}"
                )
                continue

            # Validate all choices are strings
            choices = q['choices']
            if len(choices) < 4:
                print(
                    f"Warning: Question {i} has only {len(choices)} choices, expected 4"
                )
                continue

            # Convert any non-string choices to strings
            str_choices = []
            for j, choice in enumerate(
                    choices[:4]):  # Only take first 4 choices
                if not isinstance(choice, str):
                    print(
                        f"Warning: Question {i} choice {j} is {type(choice)}, converting to string"
                    )
                    str_choices.append(str(choice))
                else:
                    str_choices.append(choice)

            # Check if choices already have letter prefixes (e.g., "A) Choice")
            if str_choices[0].strip().startswith(('A)', 'A.', 'A:')):
                # Remove the letter prefixes to match prepare_data_wmdp format
                clean_choices = []
                for choice in str_choices:
                    # Remove patterns like "A) ", "A. ", "A: "
                    choice = choice.strip()
                    if len(choice) > 2 and choice[1] in ').:' and choice[
                            0] in 'ABCD':
                        clean_choices.append(choice[2:].strip())
                    else:
                        clean_choices.append(choice)
                str_choices = clean_choices

            # Format exactly like prepare_data_wmdp
            prompt = f"""\
The following is a multiple choice question (with answer).

{q['question']}
A. {str_choices[0]}
B. {str_choices[1]}
C. {str_choices[2]}
D. {str_choices[3]}
Answer:
"""

            # Get model response
            response_idx = answer_single_question(model, tokenizer, prompt)

            # Convert index to letter
            response = ['A', 'B', 'C', 'D'][response_idx.item()]

            # Check if correct
            is_correct = response == q['answer']

            # Store result
            result = {
                'question_id': i,
                'question': q['question'],
                'choices':
                '|'.join(q['choices']),  # Join choices with | for CSV
                'correct_answer': q['answer'],
                'model_response': response,
                'is_correct': is_correct,
                'topic': q.get('topic', 'unknown'),
                'distance': q.get('distance',
                                  -1),  # Include distance for ripple analysis
                'source': q.get('source', 'unknown'),
                'model_name': model_display_name
            }

            results.append(result)

        except Exception as e:
            print(f"Error on question {i}: {e}")
            print(
                f"Question type: {type(q.get('question'))}, Choices type: {type(q.get('choices'))}"
            )
            if i == 218958:  # Debug the specific problematic question
                print(f"Question 218958 data: {q}")
            # Try to safely extract question text
            question_text = q.get('question', '')
            if isinstance(question_text, list):
                question_text = ' '.join(str(item) for item in question_text)
            else:
                question_text = str(question_text)

            results.append({
                'question_id':
                i,
                'question':
                question_text,
                'choices':
                '|'.join(str(c) for c in q.get('choices', [])),
                'correct_answer':
                str(q.get('answer', '')),
                'model_response':
                'ERROR',
                'is_correct':
                False,
                'topic':
                q.get('topic', 'unknown'),
                'distance':
                q.get('distance', -1),
                'source':
                q.get('source', 'unknown'),
                'model_name':
                model_name,
                'error':
                str(e)
            })

    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    # Calculate and print summary statistics
    accuracy = df['is_correct'].mean()
    correct = df['is_correct'].sum()
    total = len(df)

    print(f"\n{'='*50}")
    print(f"Evaluation Complete!")
    print(f"{'='*50}")
    print(f"Model: {model_display_name}")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"Results saved to: {output_csv}")

    # Also save summary stats
    summary = {
        'model_name': model_display_name,
        'dataset_path': dataset_path if not hf_dataset else f"hf:{hf_dataset}",
        'total_questions': total,
        'correct': int(correct),
        'accuracy': float(accuracy),
        'output_csv': output_csv
    }

    summary_path = Path(output_csv).with_suffix('.summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    # Clean up model from memory
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    # Delete model from cache if requested and not a base model
    if delete_after and model_display_name not in BASE_MODELS:
        print(f"\nDeleting {model_display_name} from cache...")
        delete_model_from_cache(model_id, cache_dir)
    elif delete_after and model_display_name in BASE_MODELS:
        print(f"\nKeeping base model {model_display_name} in cache")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on Ripple Bench Dataset")

    # Make dataset_path and hf_dataset mutually exclusive
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("dataset_path",
                               nargs='?',
                               help="Path to ripple bench dataset JSON file")
    dataset_group.add_argument(
        "--hf-dataset",
        help="Hugging Face dataset name (e.g., royrin/ripple-bench)")

    parser.add_argument(
        "model_name",
        help="HuggingFace model name (e.g., HuggingFaceH4/zephyr-7b-beta)")
    parser.add_argument("--output-csv",
                        required=True,
                        help="Output CSV file for results")
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help=
        f"Cache directory for HuggingFace models (default: {DEFAULT_CACHE_DIR})"
    )
    parser.add_argument(
        "--hf-cache",
        default=None,
        help="Override cache directory (deprecated, use --cache-dir instead)")
    parser.add_argument(
        "--delete-after",
        action='store_true',
        help="Delete model from cache after evaluation (keeps base models)")
    parser.add_argument("--list-models",
                        action='store_true',
                        help="List available model shortcuts and exit")

    args = parser.parse_args()

    # Handle list-models option
    if args.list_models:
        print("Available model shortcuts:")
        for name, path in MODELS.items():
            base_marker = " [BASE]" if name in BASE_MODELS else ""
            print(f"  {name}: {path}{base_marker}")
        return

    # Use hf_cache if provided, otherwise fall back to cache_dir
    cache_directory = args.hf_cache or args.cache_dir

    evaluate_model(dataset_path=args.dataset_path,
                   model_name=args.model_name,
                   output_csv=args.output_csv,
                   cache_dir=cache_directory,
                   hf_dataset=args.hf_dataset,
                   delete_after=args.delete_after)


if __name__ == "__main__":
    main()
