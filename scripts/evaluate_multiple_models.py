#!/usr/bin/env python3
"""
Evaluate multiple models on Ripple Bench Dataset and save results to CSV files.
This script runs evaluations sequentially and deletes models after evaluation to save space.

Usage:
    python evaluate_multiple_models.py --hf-dataset royrin/ripple-bench --output-dir results/
    python evaluate_multiple_models.py data/ripple_bench.json --output-dir results/
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

# Model configurations
MODELS = {
    # Base models (keep these in memory)
    "Llama-3-8b-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "zephyr-7b-beta": "HuggingFaceH4/zephyr-7b-beta",

    # LLM-GAT models (delete after evaluation)
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

    # Zephyr unlearned models (delete after evaluation)
    "zephyr-7b-elm": "baulab/elm-zephyr-7b-beta",
    "zephyr-7b-rmu": "cais/Zephyr_RMU",
}

# LLM-GAT models that have multiple checkpoints
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


def evaluate_single_model(model_name: str,
                          model_id: str,
                          questions: list,
                          output_dir: Path,
                          cache_dir: str = None,
                          delete_after: bool = False):
    """Evaluate a single model and optionally delete it after evaluation."""

    output_csv = output_dir / f"{model_name}_ripple_results.csv"

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Model ID: {model_id}")
    print(f"Output: {output_csv}")
    print(f"{'='*60}")

    # Skip if results already exist
    if output_csv.exists():
        print(f"Results already exist at {output_csv}, skipping...")
        return

    try:
        # Load model
        model, tokenizer = load_model(model_id, cache_dir)

        # Evaluate
        results = []
        for i, q in enumerate(tqdm(questions,
                                   desc=f"Evaluating {model_name}")):
            try:
                # Ensure choices is a list
                choices = q.get('choices', [])
                if not isinstance(choices, list):
                    choices = [choices]

                # Convert all choices to strings
                str_choices = []
                for choice in choices:
                    if isinstance(choice, dict):
                        str_choices.append(str(choice))
                    else:
                        str_choices.append(choice)

                # Check if choices already have letter prefixes
                if str_choices[0].strip().startswith(('A)', 'A.', 'A:')):
                    # Remove the letter prefixes
                    clean_choices = []
                    for choice in str_choices:
                        choice = choice.strip()
                        if len(choice) > 2 and choice[1] in ').:' and choice[
                                0] in 'ABCD':
                            clean_choices.append(choice[2:].strip())
                        else:
                            clean_choices.append(choice)
                    str_choices = clean_choices

                # Format prompt
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
                response = ['A', 'B', 'C', 'D'][response_idx.item()]

                # Check if correct
                is_correct = response == q['answer']

                # Store result
                result = {
                    'question_id': i,
                    'question': q['question'],
                    'choices': '|'.join(q['choices']),
                    'correct_answer': q['answer'],
                    'model_response': response,
                    'is_correct': is_correct,
                    'topic': q.get('topic', 'unknown'),
                    'distance': q.get('distance', -1),
                    'model_name': model_name
                }
                results.append(result)

            except Exception as e:
                print(f"Error on question {i}: {e}")
                results.append({
                    'question_id':
                    i,
                    'question':
                    str(q.get('question', '')),
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
                    'model_name':
                    model_name,
                    'error':
                    str(e)
                })

        # Save results
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)

        # Calculate and print summary
        accuracy = df['is_correct'].mean()
        correct = df['is_correct'].sum()
        total = len(df)

        print(f"\nResults for {model_name}:")
        print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
        print(f"Results saved to: {output_csv}")

        # Save summary
        summary = {
            'model_name': model_name,
            'model_id': model_id,
            'total_questions': total,
            'correct': int(correct),
            'accuracy': float(accuracy),
            'output_csv': str(output_csv)
        }

        summary_path = output_csv.with_suffix('.summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Clean up model from memory
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

        # Delete model from cache if requested and not a base model
        # Check both the exact name and the base name (without checkpoint suffix)
        base_name = model_name.split(
            '-ckpt')[0] if '-ckpt' in model_name else model_name
        if delete_after and base_name not in BASE_MODELS:
            delete_model_from_cache(model_id, cache_dir)

    except Exception as e:
        print(f"Failed to evaluate {model_name}: {e}")
        # Still try to clean up
        torch.cuda.empty_cache()
        gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multiple models on Ripple Bench Dataset")

    # Dataset source (mutually exclusive)
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("dataset_path",
                               nargs='?',
                               help="Path to ripple bench dataset JSON file")
    dataset_group.add_argument(
        "--hf-dataset",
        help="Hugging Face dataset name (e.g., royrin/ripple-bench)")

    parser.add_argument("--output-dir",
                        required=True,
                        help="Output directory for results")
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help=
        f"Cache directory for HuggingFace models (default: {DEFAULT_CACHE_DIR})"
    )
    parser.add_argument("--models",
                        nargs='+',
                        help="Specific models to evaluate (defaults to all)")
    parser.add_argument(
        "--model-index",
        type=int,
        help="Index of single model to evaluate (0-based, for SLURM array jobs)"
    )
    parser.add_argument(
        "--delete-after",
        action='store_true',
        help="Delete models from cache after evaluation (keeps base models)")
    parser.add_argument(
        "--all-checkpoints",
        action='store_true',
        help="Evaluate all checkpoints (1-8) for LLM-GAT models")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset once
    if args.hf_dataset:
        print(f"Loading dataset from Hugging Face: {args.hf_dataset}")
        dataset = load_dataset_from_hf(args.hf_dataset)
    else:
        print(f"Loading dataset from {args.dataset_path}")
        dataset = read_dict(args.dataset_path)

    # Extract questions
    if 'questions' in dataset:
        questions = dataset['questions']
    elif 'topics' in dataset:
        questions = []
        for topic_data in dataset['topics']:
            if 'questions' in topic_data:
                for q in topic_data['questions']:
                    if 'topic' not in q:
                        q['topic'] = topic_data.get('topic', 'unknown')
                    if 'distance' not in q:
                        q['distance'] = topic_data.get('distance', -1)
                    questions.append(q)
    else:
        raise ValueError("Cannot find questions in dataset")

    print(f"Loaded {len(questions)} questions")

    # Select models to evaluate
    if args.model_index is not None:
        # Use model index for SLURM array jobs
        model_list = list(MODELS.keys())
        if args.model_index >= len(model_list):
            raise ValueError(
                f"Model index {args.model_index} out of range. Available models: 0-{len(model_list)-1}"
            )
        selected_model = model_list[args.model_index]
        base_models = {selected_model: MODELS[selected_model]}
        print(f"Selected model at index {args.model_index}: {selected_model}")
    elif args.models:
        base_models = {k: v for k, v in MODELS.items() if k in args.models}
    else:
        base_models = MODELS.copy()

    # Expand LLM-GAT models to all checkpoints if requested
    models_to_eval = {}
    if args.all_checkpoints:
        for model_name, model_path in base_models.items():
            if model_name in LLM_GAT_MODELS:
                # Expand to all checkpoints (1-8)
                base_path = LLM_GAT_MODELS[model_name]
                for checkpoint in range(1, 9):
                    checkpoint_name = f"{model_name}-ckpt{checkpoint}"
                    checkpoint_path = f"{base_path}-{checkpoint}"
                    models_to_eval[checkpoint_name] = checkpoint_path
            else:
                # Keep non-LLM-GAT models as-is
                models_to_eval[model_name] = model_path
    else:
        models_to_eval = base_models

    print(f"\nWill evaluate {len(models_to_eval)} models:")
    for name in sorted(models_to_eval.keys()):
        print(f"  - {name}")

    # Evaluate each model
    for model_name, model_id in models_to_eval.items():
        evaluate_single_model(model_name=model_name,
                              model_id=model_id,
                              questions=questions,
                              output_dir=output_dir,
                              cache_dir=args.cache_dir,
                              delete_after=args.delete_after)

    print(f"\n{'='*60}")
    print("All evaluations complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
