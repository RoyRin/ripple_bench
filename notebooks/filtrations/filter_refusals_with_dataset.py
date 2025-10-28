#!/usr/bin/env python3
"""
Filter evaluation results by loading question text from the ripple bench dataset.
This handles the new compact CSV format that doesn't include question text.
"""

import pandas as pd
import json
from pathlib import Path
import argparse
from tqdm import tqdm
from typing import Dict


def load_question_mapping(dataset_path: str) -> Dict[int, str]:
    """
    Load the ripple bench dataset and create a mapping of assigned_question_id to question text.

    Args:
        dataset_path: Path to ripple_bench_dataset.json

    Returns:
        Dictionary mapping assigned_question_id to question text
    """
    print(f"Loading questions from: {dataset_path}")

    with open(dataset_path, 'r') as f:
        data = json.load(f)

    question_map = {}

    # Check if questions are in 'topics' structure
    if 'topics' in data:
        for topic_data in data['topics']:
            if 'questions' in topic_data:
                for q in topic_data['questions']:
                    if 'assigned_question_id' in q and 'question' in q:
                        question_map[q['assigned_question_id']] = q['question']

    # Also check for questions in other structures
    if 'questions' in data:
        for q in data['questions']:
            if 'assigned_question_id' in q and 'question' in q:
                question_map[q['assigned_question_id']] = q['question']

    print(f"  Loaded {len(question_map):,} unique questions")
    return question_map


def should_filter_question(question_text: str) -> tuple[bool, str]:
    """
    Check if a question should be filtered out.

    Args:
        question_text: The question text to check

    Returns:
        Tuple of (should_filter, reason)
    """
    question_lower = question_text.lower()

    # Check for Wikipedia errors
    if "Article not found in local Wikipedia" in question_text:
        return True, "wikipedia_not_found"

    if " 404" in question_text:
        return True, "404_error"

    if "wikipedia" in question_lower:
        return True, "wikipedia_mention"

    # Check for refusal phrases
    refusal_phrases = [
        "i can't", "i cannot", "i won't", "harmful", "bioweapon",
        "i'm not able", "i am not able", "dangerous", "unethical",
        "i shouldn't", "i should not", "article not found"
    ]

    for phrase in refusal_phrases:
        if phrase in question_lower:
            return True, "refusal"

    return False, "ok"


def filter_results_with_questions(input_csv: str,
                                  output_csv: str,
                                  question_map: Dict[int, str],
                                  verbose: bool = False):
    """
    Filter a single CSV file using question text from the dataset.
    """
    csv_name = Path(input_csv).name
    print(f"\n  Processing: {csv_name}")

    df = pd.read_csv(input_csv)
    original_count = len(df)

    if 'assigned_question_id' not in df.columns:
        print(
            f"    WARNING: No 'assigned_question_id' column found. Cannot filter."
        )
        df.to_csv(output_csv, index=False)
        return df

    # Track filtering statistics
    filter_counts = {
        'wikipedia_not_found': 0,
        '404_error': 0,
        'wikipedia_mention': 0,
        'refusal': 0,
        'no_question_found': 0
    }

    # Create filter mask
    filter_mask = []

    for idx, row in df.iterrows():
        q_id = row['assigned_question_id']

        # Check if we have the question text
        if q_id not in question_map:
            filter_mask.append(
                True)  # Keep rows where we can't find the question
            filter_counts['no_question_found'] += 1
            if verbose and filter_counts['no_question_found'] <= 5:
                print(
                    f"    WARNING: No question found for assigned_question_id {q_id}"
                )
            continue

        question_text = question_map[q_id]
        should_filter, reason = should_filter_question(question_text)

        if should_filter:
            filter_counts[reason] += 1
            filter_mask.append(False)  # Remove this row
        else:
            filter_mask.append(True)  # Keep this row

    # Apply filter
    filtered_df = df[filter_mask]

    # Save filtered results
    filtered_df.to_csv(output_csv, index=False)

    filtered_count = len(filtered_df)
    removed_count = original_count - filtered_count
    retention_rate = (filtered_count / original_count *
                      100) if original_count > 0 else 0

    print(
        f"    Rows: {original_count:,} → {filtered_count:,} ({retention_rate:.1f}% retained)"
    )
    print(f"    Removed {removed_count:,} rows:")
    for reason, count in filter_counts.items():
        if count > 0:
            print(f"      - {reason}: {count:,}")

    if 'is_correct' in filtered_df.columns and len(filtered_df) > 0:
        orig_accuracy = df['is_correct'].mean() * 100
        new_accuracy = filtered_df['is_correct'].mean() * 100
        print(f"    Accuracy: {orig_accuracy:.2f}% → {new_accuracy:.2f}%")

    return filtered_df


def process_directory(input_dir: str,
                      output_dir: str,
                      dataset_path: str,
                      suffix: str = "_filtered"):
    """
    Process all CSV files in a directory.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Load question mapping once
    question_map = load_question_mapping(dataset_path)

    # Find all CSV files
    csv_files = list(input_path.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    print(f"\nFound {len(csv_files)} CSV files to process")
    print("=" * 60)

    # Track overall statistics
    total_original = 0
    total_filtered = 0

    # Process each file
    for csv_file in sorted(csv_files):
        # Create output filename
        output_name = csv_file.stem + suffix + ".csv"
        output_file = output_path / output_name

        try:
            original_df = pd.read_csv(csv_file)
            total_original += len(original_df)

            filtered_df = filter_results_with_questions(
                str(csv_file), str(output_file), question_map)
            total_filtered += len(filtered_df)

        except Exception as e:
            print(f"  ERROR: Failed to process {csv_file.name}: {e}")
            continue

    # Print summary
    print("\n" + "=" * 60)
    print(f"Filtering complete!")
    if total_original > 0:
        print(
            f"Total rows: {total_original:,} → {total_filtered:,} ({total_filtered/total_original*100:.1f}% retained)"
        )
    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=
        "Filter evaluation results for refusals using the ripple bench dataset"
    )

    # Required arguments
    parser.add_argument("input_dir",
                        help="Directory containing CSV files to filter")
    parser.add_argument("output_dir",
                        help="Directory to save filtered CSV files")
    parser.add_argument("--dataset",
                        required=True,
                        help="Path to ripple_bench_dataset.json")

    # Optional arguments
    parser.add_argument(
        "--suffix",
        default="_filtered",
        help="Suffix for filtered files (default: '_filtered')")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Show detailed warnings")

    args = parser.parse_args()

    # Process all files in directory
    process_directory(args.input_dir, args.output_dir, args.dataset,
                      args.suffix)


if __name__ == "__main__":
    main()
