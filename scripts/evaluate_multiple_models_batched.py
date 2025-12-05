#!/usr/bin/env python3
"""
Evaluate multiple models on Ripple Bench Dataset with BATCHED inference.
This script uses batch processing for faster evaluation compared to the sequential version.

Usage:
    python evaluate_multiple_models_batched.py --hf-dataset royrin/ripple-bench --output-dir results/ --batch-size 32
    python evaluate_multiple_models_batched.py data/ripple_bench.json --output-dir results/ --batch-size 16
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
import re
import traceback

from ripple_bench.metrics import answer_single_question
from ripple_bench.utils import read_dict
from ripple_bench.config import DEFAULT_CACHE_DIR

# Model configurations
MODELS = {
    # Base models (keep these in memory)
    "Llama-3-8b-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "zephyr-7b-beta": "HuggingFaceH4/zephyr-7b-beta",
    # LLM-GAT models (delete after evaluation)
    "llama-3-8b-instruct-graddiff": "LLM-GAT/llama-3-8b-instruct-graddiff-checkpoint-8",
    "llama-3-8b-instruct-elm": "LLM-GAT/llama-3-8b-instruct-elm-checkpoint-8",
    "llama-3-8b-instruct-pbj": "LLM-GAT/llama-3-8b-instruct-pbj-checkpoint-8",
    "llama-3-8b-instruct-tar": "LLM-GAT/llama-3-8b-instruct-tar-checkpoint-8",
    "llama-3-8b-instruct-rmu": "LLM-GAT/llama-3-8b-instruct-rmu-checkpoint-8",
    "llama-3-8b-instruct-rmu-lat": "LLM-GAT/llama-3-8b-instruct-rmu-lat-checkpoint-8",
    "llama-3-8b-instruct-repnoise": "LLM-GAT/llama-3-8b-instruct-repnoise-checkpoint-8",
    "llama-3-8b-instruct-rr": "LLM-GAT/llama-3-8b-instruct-rr-checkpoint-8",
    # Zephyr unlearned models (delete after evaluation)
    "zephyr-7b-elm": "baulab/elm-zephyr-7b-beta",
    "zephyr-7b-rmu": "cais/Zephyr_RMU",
}

# LLM-GAT models that have multiple checkpoints
LLM_GAT_MODELS = {
    "llama-3-8b-instruct-graddiff": "LLM-GAT/llama-3-8b-instruct-graddiff-checkpoint",
    "llama-3-8b-instruct-elm": "LLM-GAT/llama-3-8b-instruct-elm-checkpoint",
    "llama-3-8b-instruct-pbj": "LLM-GAT/llama-3-8b-instruct-pbj-checkpoint",
    "llama-3-8b-instruct-tar": "LLM-GAT/llama-3-8b-instruct-tar-checkpoint",
    "llama-3-8b-instruct-rmu": "LLM-GAT/llama-3-8b-instruct-rmu-checkpoint",
    "llama-3-8b-instruct-rmu-lat": "LLM-GAT/llama-3-8b-instruct-rmu-lat-checkpoint",
    "llama-3-8b-instruct-repnoise": "LLM-GAT/llama-3-8b-instruct-repnoise-checkpoint",
    "llama-3-8b-instruct-rr": "LLM-GAT/llama-3-8b-instruct-rr-checkpoint",
}

# Base models to keep in cache
BASE_MODELS = {"Llama-3-8b-Instruct", "zephyr-7b-beta"}


def discover_models_from_cache(cache_dir: str) -> Dict[str, str]:
    """Discover LLM-GAT models and base model from the cache directory."""
    cache_path = Path(cache_dir)
    discovered_models = {}

    if not cache_path.exists():
        print(f"Cache directory {cache_dir} does not exist")
        return discovered_models

    # Add base model first (index 0)
    base_model_path = cache_path / "models--meta-llama--Meta-Llama-3-8B-Instruct"
    if base_model_path.exists():
        discovered_models["Llama-3-8b-Instruct"] = "meta-llama/Meta-Llama-3-8B-Instruct"
        print("Found base model: Llama-3-8b-Instruct")

    # Look for models--LLM-GAT--* directories
    for model_dir in cache_path.glob("models--LLM-GAT--*"):
        # Extract model name from directory name
        # Format: models--LLM-GAT--llama-3-8b-instruct-elm-checkpoint-1
        model_name = model_dir.name.replace("models--LLM-GAT--", "")

        # Convert to HuggingFace model ID format
        # llama-3-8b-instruct-elm-checkpoint-1 -> LLM-GAT/llama-3-8b-instruct-elm-checkpoint-1
        model_id = f"LLM-GAT/{model_name}"

        # Create a clean display name
        # Remove checkpoint number for cleaner naming
        display_name = model_name.replace("-checkpoint-", "-ckpt")

        discovered_models[display_name] = model_id

    print(f"Discovered {len(discovered_models)} models from cache:")
    for i, name in enumerate(sorted(discovered_models.keys())):
        print(f"  - {i}: {name}")

    return discovered_models


def load_dataset_from_hf(dataset_name: str):
    """Load Ripple Bench dataset from Hugging Face."""
    print(f"Loading dataset from Hugging Face: {dataset_name}")

    # Load the dataset
    dataset = load_dataset(dataset_name)

    # The dataset should have a 'train' split with the questions
    if "train" in dataset:
        data = dataset["train"]
    else:
        # If no train split, use the first available split
        data = dataset[list(dataset.keys())[0]]

    # Convert to the expected format
    questions = []
    for item in data:
        # Each item should have the question structure
        question = {
            "question": item.get("question", ""),
            "choices": item.get("choices", []),
            "answer": item.get("answer", ""),
            "original_topics": item.get("original_topics", []),
            "topic": item.get("topic", "unknown"),
            "distance": item.get("distance", -1),
            "id": item.get("id", ""),
        }
        questions.append(question)

    return {"questions": questions}


def load_model(model_id: str, device: str, cache_dir: str = None):
    """Load a model from HuggingFace Hub."""
    print(f"Loading model: {model_id} on {device}")
    dtype = torch.float32 if device == "cpu" else torch.float16

    use_device_map = device.startswith("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, cache_dir=cache_dir, device_map="auto" if use_device_map else None
    )

    if not use_device_map:
        model = model.to(device)

    model.requires_grad_(False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, cache_dir=cache_dir)
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


def save_results_and_cleanup(
    results: list,
    model_name: str,
    model_id: str,
    output_csv: Path,
    model,
    tokenizer,
    cache_dir: str = None,
    delete_after: bool = False,
    extra_stats: dict = None,
):
    """Save evaluation results to CSV, print summary, and cleanup.

    Args:
        results: List of result dictionaries
        model_name: Name of the model
        model_id: HuggingFace model ID
        output_csv: Path to save CSV
        model: Model to cleanup
        tokenizer: Tokenizer to cleanup
        cache_dir: Cache directory for deletion
        delete_after: Whether to delete model from cache
        extra_stats: Extra statistics to include in summary (e.g., cache stats)
    """
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    print(df)

    # Calculate and print summary
    accuracy = df["is_correct"].mean()
    correct = df["is_correct"].sum()
    total = len(df)

    print(f"\nResults for {model_name}:")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")

    # Print extra stats if provided
    if extra_stats:
        for key, value in extra_stats.items():
            print(f"{key}: {value}")

    print(f"Results saved to: {output_csv}")

    # Save summary
    summary = {
        "model_name": model_name,
        "model_id": model_id,
        "total_questions": total,
        "correct": int(correct),
        "accuracy": float(accuracy),
        "output_csv": str(output_csv),
    }

    # Add extra stats to summary
    if extra_stats:
        summary.update(extra_stats)

    summary_path = output_csv.with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Clean up model from memory
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    # Delete model from cache if requested and not a base model
    base_name = model_name.split("-ckpt")[0] if "-ckpt" in model_name else model_name
    if delete_after and base_name not in BASE_MODELS:
        delete_model_from_cache(model_id, cache_dir)


def get_model_response(model, tokenizer, question: dict) -> tuple[str, bool]:
    """Get model response for a single question.

    Returns:
        tuple: (response letter, is_correct)
    """
    # Ensure choices is a list
    choices = question.get("choices", [])
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
    if str_choices[0].strip().startswith(("A)", "A.", "A:")):
        # Remove the letter prefixes
        clean_choices = []
        for choice in str_choices:
            choice = choice.strip()
            if len(choice) > 2 and choice[1] in ").:" and choice[0] in "ABCD":
                clean_choices.append(choice[2:].strip())
            else:
                clean_choices.append(choice)
        str_choices = clean_choices

    # Format prompt
    prompt = f"""\
The following is a multiple choice question (with answer).

{question["question"]}
A. {str_choices[0]}
B. {str_choices[1]}
C. {str_choices[2]}
D. {str_choices[3]}
Answer:
"""

    # Get model response
    response_idx = answer_single_question(model, tokenizer, prompt)
    response = ["A", "B", "C", "D"][response_idx.item()]

    # Check if correct
    is_correct = response == question["answer"]

    return response, is_correct


def format_question_prompt(question: dict) -> tuple[str, list[str]]:
    """Format a single question into a prompt.

    Returns:
        tuple: (prompt string, cleaned choices list)
    """
    # Ensure choices is a list
    choices = question.get("choices", [])
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
    if str_choices[0].strip().startswith(("A)", "A.", "A:")):
        # Remove the letter prefixes
        clean_choices = []
        for choice in str_choices:
            choice = choice.strip()
            if len(choice) > 2 and choice[1] in ").:" and choice[0] in "ABCD":
                clean_choices.append(choice[2:].strip())
            else:
                clean_choices.append(choice)
        str_choices = clean_choices

    # Format prompt
    prompt = f"""\
The following is a multiple choice question (with answer).

{question["question"]}
A. {str_choices[0]}
B. {str_choices[1]}
C. {str_choices[2]}
D. {str_choices[3]}
Answer:
"""
    return prompt, str_choices


def get_model_responses_batch(model, tokenizer, questions: list[dict]) -> list[tuple[str, bool]]:
    """Get model responses for a batch of questions.

    Returns:
        list of tuples: [(response letter, is_correct), ...]
    """
    from ripple_bench.metrics import answer_batch_questions

    # Format all prompts
    prompts = []
    for q in questions:
        prompt, _ = format_question_prompt(q)
        prompts.append(prompt)

    # Get batch responses
    response_indices = answer_batch_questions(model, tokenizer, prompts)

    # Convert to letters and check correctness
    results = []
    for i, q in enumerate(questions):
        response = ["A", "B", "C", "D"][response_indices[i].item()]
        is_correct = response == q["answer"]
        results.append((response, is_correct))

    return results


def evaluate_single_model_hf(
    model_name: str,
    model_id: str,
    questions: list,
    output_dir: Path,
    device: str,
    cache_dir: str = None,
    delete_after: bool = False,
    batch_size: int = 32,
):
    """Evaluate a single model on HF dataset format with batching.

    HF format has original_topics as a list. Questions are evaluated in batches,
    but each question creates one CSV row per original_topic.
    """
    output_csv = output_dir / f"{model_name}_ripple_results.csv"

    print(f"\n{'=' * 60}")
    print(f"Evaluating: {model_name} (HF format, batch_size={batch_size})")
    print(f"Model ID: {model_id}")
    print(f"Output: {output_csv}")
    print(f"{'=' * 60}")

    if output_csv.exists():
        print(f"Results already exist at {output_csv}, skipping...")
        return

    try:
        model, tokenizer = load_model(model_id, device, cache_dir)
        results = []

        # Process questions in batches
        for batch_start in tqdm(range(0, len(questions), batch_size), desc=f"Evaluating {model_name}"):
            batch_end = min(batch_start + batch_size, len(questions))
            batch_questions = questions[batch_start:batch_end]

            try:
                # Evaluate batch
                batch_responses = get_model_responses_batch(model, tokenizer, batch_questions)

                # Process each question in the batch
                for i, (q, (response, is_correct)) in enumerate(zip(batch_questions, batch_responses)):
                    question_id = batch_start + i

                    # Get original_topics list
                    original_topics = q.get("original_topics", [])
                    if not isinstance(original_topics, list):
                        original_topics = [original_topics]

                    # Create one row per original topic
                    for og_topic in original_topics:
                        result = {
                            "question_id": question_id,
                            "model_response": response,
                            "is_correct": is_correct,
                            "topic": q.get("topic", "unknown"),
                            "original_topic": og_topic.get("topic", "unknown") if isinstance(og_topic, dict) else str(og_topic),
                            "distance": og_topic.get("distance", -1) if isinstance(og_topic, dict) else -1,
                            "model_name": model_name,
                        }
                        results.append(result)

            except Exception as e:
                print(f"Error on batch starting at {batch_start}: {e}")
                # For errors, create error rows for all questions in the batch
                for i, q in enumerate(batch_questions):
                    question_id = batch_start + i
                    original_topics = q.get("original_topics", [{"topic": "unknown", "distance": -1}])
                    for og_topic in original_topics:
                        results.append(
                            {
                                "question_id": question_id,
                                "model_response": "ERROR",
                                "is_correct": False,
                                "topic": q.get("topic", "unknown"),
                                "original_topic": og_topic.get("topic", "unknown") if isinstance(og_topic, dict) else str(og_topic),
                                "distance": og_topic.get("distance", -1) if isinstance(og_topic, dict) else -1,
                                "model_name": model_name,
                                "error": str(e),
                            }
                        )

        # Extra stats for HF format
        hf_stats = {
            "unique_questions": len(questions),
            "total_csv_rows": len(results),
        }

        print(f"\nHF Dataset Statistics:")
        print(f"  Unique questions evaluated: {len(questions)}")
        print(f"  Total CSV rows (one per original_topic): {len(results)}")

        # Save results and cleanup
        save_results_and_cleanup(
            results=results,
            model_name=model_name,
            model_id=model_id,
            output_csv=output_csv,
            model=model,
            tokenizer=tokenizer,
            cache_dir=cache_dir,
            delete_after=delete_after,
            extra_stats=hf_stats,
        )

    except Exception as e:
        print(f"Failed to evaluate {model_name}: {e}")
        traceback.print_exc()
        torch.cuda.empty_cache()
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple models on Ripple Bench Dataset")

    # Dataset source (HF only for batched version)
    parser.add_argument("--hf-dataset", required=True, help="Hugging Face dataset name (e.g., RippleBench/ripple-bench)")

    parser.add_argument("--output-dir", help="Output directory for results")
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help=f"Cache directory for HuggingFace models (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use for model inference (e.g., 'cuda:0', 'cuda:1', 'cpu'). Defaults to 'cuda:0' if available, else 'cpu'",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    parser.add_argument("--models", nargs="+", help="Specific models to evaluate (defaults to all)")
    parser.add_argument(
        "--model-index", type=int, help="Index of single model to evaluate (0-based, for SLURM array jobs)"
    )
    parser.add_argument(
        "--delete-after", action="store_true", help="Delete models from cache after evaluation (keeps base models)"
    )
    parser.add_argument(
        "--all-checkpoints", action="store_true", help="Evaluate all checkpoints (1-8) for LLM-GAT models"
    )
    parser.add_argument(
        "--discover-cache", action="store_true", help="Discover and evaluate all LLM-GAT models from cache directory"
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List all available models and exit"
    )

    args = parser.parse_args()

    # Handle --list-models
    if args.list_models:
        print("Available predefined models:")
        print("=" * 60)
        for i, (name, model_id) in enumerate(sorted(MODELS.items())):
            base_or_unlearned = "BASE" if name in BASE_MODELS else "UNLEARNED"
            print(f"{i:2d}. {name:40s} [{base_or_unlearned}]")
            print(f"    {model_id}")

        print(f"\n{'=' * 60}")
        print(f"Total: {len(MODELS)} models")
        print(f"  - Base models: {len(BASE_MODELS)}")
        print(f"  - Unlearned models: {len(MODELS) - len(BASE_MODELS)}")

        print(f"\n{'=' * 60}")
        print("LLM-GAT models with multiple checkpoints:")
        print("(Use --all-checkpoints to evaluate all 8 checkpoints)")
        print("=" * 60)
        for name, base_path in sorted(LLM_GAT_MODELS.items()):
            print(f"  {name}")
            print(f"    Checkpoints: {base_path}-1 through {base_path}-8")

        return

    # Validate required arguments when not listing models
    if not args.output_dir:
        parser.error("--output-dir is required")

    # Set HuggingFace cache environment variables
    cache_vars = ["HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"]
    for var in cache_vars:
        old_value = os.environ.get(var)
        os.environ[var] = args.cache_dir
        if old_value:
            print(f"Updated {var}: {old_value} -> {args.cache_dir}")
        else:
            print(f"Set {var}: {args.cache_dir}")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load HF dataset
    print(f"Loading dataset from Hugging Face: {args.hf_dataset}")
    dataset = load_dataset_from_hf(args.hf_dataset)
    questions = dataset["questions"]

    print(f"Loaded {len(questions)} questions")

    # Select models to evaluate
    if args.discover_cache:
        # Discover models from cache directory
        discovered_models = discover_models_from_cache(args.cache_dir)
        if not discovered_models:
            print("No LLM-GAT models found in cache directory")
            return

        if args.model_index is not None:
            # Use model index for SLURM array jobs
            model_list = list(discovered_models.keys())
            if args.model_index >= len(model_list):
                raise ValueError(
                    f"Model index {args.model_index} out of range. Available models: 0-{len(model_list) - 1}"
                )
            selected_model = model_list[args.model_index]
            base_models = {selected_model: discovered_models[selected_model]}
            print(f"Selected model at index {args.model_index}: {selected_model}")
        elif args.models:
            base_models = {k: v for k, v in discovered_models.items() if k in args.models}
        else:
            base_models = discovered_models.copy()
    else:
        # Use predefined models
        if args.model_index is not None:
            # Use model index for SLURM array jobs
            model_list = list(MODELS.keys())
            if args.model_index >= len(model_list):
                raise ValueError(
                    f"Model index {args.model_index} out of range. Available models: 0-{len(model_list) - 1}"
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
    if args.all_checkpoints and not args.discover_cache:
        # Only expand checkpoints for predefined models, not discovered ones
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

    # Evaluate each model with batching
    for model_name, model_id in models_to_eval.items():
        evaluate_single_model_hf(
            model_name=model_name,
            model_id=model_id,
            questions=questions,
            output_dir=output_dir,
            device=args.device,
            cache_dir=args.cache_dir,
            delete_after=args.delete_after,
            batch_size=args.batch_size,
        )

    print(f"\n{'=' * 60}")
    print("All evaluations complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
