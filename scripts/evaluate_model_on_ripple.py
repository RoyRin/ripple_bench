#!/usr/bin/env python3
"""
Evaluate a single model on Ripple Bench Dataset and save results to CSV

Usage:
    python evaluate_model_on_ripple.py <dataset_path> <model_name> --output-csv <output.csv>
    
Example:
    python evaluate_model_on_ripple.py data/ripple_bench.json HuggingFaceH4/zephyr-7b-beta --output-csv zephyr_base_results.csv
    python evaluate_model_on_ripple.py data/ripple_bench.json baulab/elm-zephyr-7b-beta --output-csv zephyr_elm_results.csv
"""

import json
import argparse
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ripple_bench.metrics import answer_single_question
from ripple_bench.utils import read_dict


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
                   cache_dir: str = None):
    """Evaluate a model on ripple bench dataset and save results to CSV."""

    # Load dataset
    print(f"Loading dataset from {dataset_path}")
    dataset = read_dict(dataset_path)
    questions = dataset['questions']
    print(f"Loaded {len(questions)} questions")

    # Load model
    model, tokenizer = load_model(model_name, cache_dir)

    # Evaluate
    print(f"\nEvaluating {model_name}...")
    results = []

    for i, q in enumerate(tqdm(questions, desc="Evaluating")):
        try:
            # Get model response
            response = answer_single_question(q['question'], q['choices'],
                                              model, tokenizer)

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
                'source': q.get('source', 'unknown'),
                'model_name': model_name
            }

            results.append(result)

        except Exception as e:
            print(f"Error on question {i}: {e}")
            results.append({
                'question_id': i,
                'question': q['question'],
                'choices': '|'.join(q['choices']),
                'correct_answer': q['answer'],
                'model_response': 'ERROR',
                'is_correct': False,
                'topic': q.get('topic', 'unknown'),
                'source': q.get('source', 'unknown'),
                'model_name': model_name,
                'error': str(e)
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
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"Results saved to: {output_csv}")

    # Also save summary stats
    summary = {
        'model_name': model_name,
        'dataset_path': dataset_path,
        'total_questions': total,
        'correct': int(correct),
        'accuracy': float(accuracy),
        'output_csv': output_csv
    }

    summary_path = Path(output_csv).with_suffix('.summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on Ripple Bench Dataset")
    parser.add_argument("dataset_path",
                        help="Path to ripple bench dataset JSON file")
    parser.add_argument(
        "model_name",
        help="HuggingFace model name (e.g., HuggingFaceH4/zephyr-7b-beta)")
    parser.add_argument("--output-csv",
                        required=True,
                        help="Output CSV file for results")
    parser.add_argument("--cache-dir",
                        default=None,
                        help="Cache directory for HuggingFace models")

    args = parser.parse_args()

    evaluate_model(dataset_path=args.dataset_path,
                   model_name=args.model_name,
                   output_csv=args.output_csv,
                   cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
