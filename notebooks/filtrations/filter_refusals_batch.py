#!/usr/bin/env python3
"""
Batch filter evaluation results to remove refusals and Wikipedia errors.
Works with CSV files that contain the full question text.
"""

import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm


def filter_refusals_and_errors(input_csv: str,
                               output_csv: str,
                               verbose: bool = False):
    """
    Filter a single CSV file to remove refusals and Wikipedia errors.

    This filters out:
    1. Questions containing "Article not found in local Wikipedia"
    2. Questions containing " 404"
    3. Questions containing "wikipedia" (case insensitive)
    4. Questions with refusal phrases like "I can't", "I cannot", "I won't", "harmful", "bioweapon"
    """
    print(f"\n  Loading: {Path(input_csv).name}")
    df = pd.read_csv(input_csv)
    original_count = len(df)

    if 'question' not in df.columns:
        print(
            f"    WARNING: No 'question' column found. Skipping refusal filtering."
        )
        df.to_csv(output_csv, index=False)
        return df

    # Count different types of filtered questions
    wikipedia_errors = 0
    refusals = 0

    # Filter function
    def should_filter(row):
        nonlocal wikipedia_errors, refusals

        question = str(row['question'])
        question_lower = question.lower()

        # Check for Wikipedia errors
        if ("Article not found in local Wikipedia" in question
                or " 404" in question or "wikipedia" in question_lower):
            wikipedia_errors += 1
            return True

        # Check for refusal phrases
        refusal_phrases = [
            "i can't", "i cannot", "i won't", "harmful", "bioweapon",
            "i'm not able", "i am not able", "dangerous", "unethical",
            "i shouldn't", "i should not", "article not found"
        ]

        if any(phrase in question_lower for phrase in refusal_phrases):
            refusals += 1
            return True

        return False

    # Apply filter
    mask = df.apply(should_filter, axis=1)
    filtered_df = df[~mask]

    # Save filtered results
    filtered_df.to_csv(output_csv, index=False)

    filtered_count = len(filtered_df)
    retention_rate = (filtered_count / original_count *
                      100) if original_count > 0 else 0

    print(
        f"    Filtered: {original_count:,} → {filtered_count:,} ({retention_rate:.1f}% retained)"
    )
    print(
        f"    Removed: Wikipedia errors: {wikipedia_errors:,}, Refusals: {refusals:,}"
    )

    if 'is_correct' in filtered_df.columns:
        orig_accuracy = df['is_correct'].mean() * 100
        new_accuracy = filtered_df['is_correct'].mean() * 100
        print(f"    Accuracy: {orig_accuracy:.2f}% → {new_accuracy:.2f}%")

    return filtered_df


def process_directory(input_dir: str,
                      output_dir: str,
                      suffix: str = "_filtered"):
    """
    Process all CSV files in a directory.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all CSV files
    csv_files = list(input_path.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    print(f"Found {len(csv_files)} CSV files to process")
    print("=" * 60)

    # Track overall statistics
    total_original = 0
    total_filtered = 0

    # Process each file
    for csv_file in tqdm(sorted(csv_files), desc="Processing files"):
        # Create output filename
        output_name = csv_file.stem + suffix + ".csv"
        output_file = output_path / output_name

        try:
            df = filter_refusals_and_errors(str(csv_file), str(output_file))
            total_original += len(pd.read_csv(csv_file))
            total_filtered += len(df)
        except Exception as e:
            print(f"  ERROR: Failed to process {csv_file.name}: {e}")
            continue

    # Print summary
    print("\n" + "=" * 60)
    print(f"Filtering complete!")
    print(
        f"Total rows: {total_original:,} → {total_filtered:,} ({total_filtered/total_original*100:.1f}% retained)"
    )
    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch filter evaluation results for refusals and errors")

    # Required arguments
    parser.add_argument("input_dir",
                        help="Directory containing CSV files to filter")
    parser.add_argument("output_dir",
                        help="Directory to save filtered CSV files")

    # Optional arguments
    parser.add_argument(
        "--suffix",
        default="_filtered",
        help="Suffix for filtered files (default: '_filtered')")

    args = parser.parse_args()

    # Process all files in directory
    process_directory(args.input_dir, args.output_dir, args.suffix)


if __name__ == "__main__":
    main()
