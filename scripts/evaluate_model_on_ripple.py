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
    model, tokenizer = load_model(model_name, cache_dir)

    # Evaluate
    print(f"\nEvaluating {model_name}...")
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
                'model_name': model_name
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
    parser.add_argument(
        "--hf-cache",
        default=None,
        help=
        "HuggingFace cache directory (e.g., /n/netscratch/vadhan_lab/Lab/rrinberg/HF_cache)"
    )

    args = parser.parse_args()

    # Use hf_cache if provided, otherwise fall back to cache_dir
    cache_directory = args.hf_cache or args.cache_dir

    evaluate_model(dataset_path=args.dataset_path,
                   model_name=args.model_name,
                   output_csv=args.output_csv,
                   cache_dir=cache_directory)


if __name__ == "__main__":
    main()
