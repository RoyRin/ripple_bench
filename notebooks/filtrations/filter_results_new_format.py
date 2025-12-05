#!/usr/bin/env python3
"""
Filter evaluation results in the new compact format.
The new format only has: question_id, assigned_question_id, model_response, is_correct, topic, original_topic, distance, model_name

This script can filter based on:
1. Topic quality (using the neighbor quality analysis results)
2. Distance ranges
3. Model responses (e.g., removing ERROR responses)
"""

import pandas as pd
import json
from pathlib import Path
import argparse
from typing import Set, Optional


def load_valid_topics(neighbor_quality_file: str,
                      bio_chem_threshold: int = 13,
                      biosafety_threshold: int = 7,
                      any_safety_threshold: int = 15) -> Set[str]:
    """
    Load valid topics based on neighbor quality analysis.

    Args:
        neighbor_quality_file: Path to the neighbor quality JSON file
        bio_chem_threshold: Topics with <= this many non-relevant bio/chem neighbors are kept
        biosafety_threshold: Topics with <= this many non-relevant biosafety neighbors are kept
        any_safety_threshold: Topics with <= this many non-relevant safety neighbors are kept

    Returns:
        Set of valid topic names
    """
    with open(neighbor_quality_file, 'r') as f:
        quality_data = json.load(f)

    valid_topics = set()

    for topic, scores in quality_data.items():
        # Check if topic passes any of the thresholds
        if (scores.get('bio_chem', [float('inf')])[0] <= bio_chem_threshold
                or scores.get('biosafety', [float('inf')])[0]
                <= biosafety_threshold or scores.get(
                    'any_safety', [float('inf')])[0] <= any_safety_threshold):
            valid_topics.add(topic)

    print(
        f"Loaded {len(valid_topics)} valid topics from neighbor quality analysis"
    )
    return valid_topics


def filter_results(input_csv: str,
                   output_csv: str,
                   valid_topics: Optional[Set[str]] = None,
                   min_distance: Optional[int] = None,
                   max_distance: Optional[int] = None,
                   remove_errors: bool = True,
                   filter_by_original_topic: bool = False) -> pd.DataFrame:
    """
    Filter evaluation results based on various criteria.

    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output filtered CSV file
        valid_topics: Set of valid topics to keep (if None, no topic filtering)
        min_distance: Minimum distance to include (if None, no min filtering)
        max_distance: Maximum distance to include (if None, no max filtering)
        remove_errors: Whether to remove rows with model_response == 'ERROR'
        filter_by_original_topic: If True, filter by original_topic instead of topic

    Returns:
        Filtered DataFrame
    """
    print(f"\nLoading results from: {input_csv}")
    df = pd.read_csv(input_csv)
    original_count = len(df)
    print(f"  Original rows: {original_count:,}")

    # Remove ERROR responses if requested
    if remove_errors and 'model_response' in df.columns:
        error_count = (df['model_response'] == 'ERROR').sum()
        if error_count > 0:
            df = df[df['model_response'] != 'ERROR']
            print(f"  Removed {error_count:,} ERROR responses")

    # Filter by distance range
    if min_distance is not None and 'distance' in df.columns:
        before = len(df)
        df = df[df['distance'] >= min_distance]
        print(
            f"  Filtered by min_distance >= {min_distance}: {before:,} → {len(df):,}"
        )

    if max_distance is not None and 'distance' in df.columns:
        before = len(df)
        df = df[df['distance'] <= max_distance]
        print(
            f"  Filtered by max_distance <= {max_distance}: {before:,} → {len(df):,}"
        )

    # Filter by valid topics
    if valid_topics is not None:
        topic_col = 'original_topic' if filter_by_original_topic else 'topic'

        if topic_col not in df.columns:
            print(
                f"  WARNING: Column '{topic_col}' not found in CSV. Skipping topic filtering."
            )
        else:
            before = len(df)
            unique_topics_before = df[topic_col].nunique()

            # Filter to valid topics
            df = df[df[topic_col].isin(valid_topics)]

            unique_topics_after = df[topic_col].nunique()
            print(f"  Filtered by {topic_col}: {before:,} → {len(df):,} rows")
            print(
                f"    Topics: {unique_topics_before:,} → {unique_topics_after:,}"
            )

    # Save filtered results
    df.to_csv(output_csv, index=False)

    final_count = len(df)
    retention_rate = (final_count / original_count *
                      100) if original_count > 0 else 0

    print(f"\nFinal results:")
    print(
        f"  Rows: {original_count:,} → {final_count:,} ({retention_rate:.1f}% retained)"
    )
    print(f"  Saved to: {output_csv}")

    # Print summary statistics
    if 'is_correct' in df.columns:
        accuracy = df['is_correct'].mean() * 100
        print(f"  Accuracy: {accuracy:.2f}%")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Filter evaluation results in new format")

    # Required arguments
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("output_csv", help="Path to output filtered CSV file")

    # Optional filtering arguments
    parser.add_argument(
        "--neighbor-quality",
        help="Path to neighbor quality JSON file for topic filtering")
    parser.add_argument(
        "--bio-chem-threshold",
        type=int,
        default=13,
        help="Max non-relevant bio/chem neighbors (default: 13)")
    parser.add_argument(
        "--biosafety-threshold",
        type=int,
        default=7,
        help="Max non-relevant biosafety neighbors (default: 7)")
    parser.add_argument("--any-safety-threshold",
                        type=int,
                        default=15,
                        help="Max non-relevant safety neighbors (default: 15)")

    parser.add_argument("--min-distance",
                        type=int,
                        help="Minimum distance to include")
    parser.add_argument("--max-distance",
                        type=int,
                        help="Maximum distance to include")

    parser.add_argument("--keep-errors",
                        action="store_true",
                        help="Keep ERROR responses (default: remove them)")
    parser.add_argument(
        "--filter-by-original-topic",
        action="store_true",
        help="Filter by original_topic instead of topic column")

    args = parser.parse_args()

    # Load valid topics if neighbor quality file provided
    valid_topics = None
    if args.neighbor_quality:
        valid_topics = load_valid_topics(args.neighbor_quality,
                                         args.bio_chem_threshold,
                                         args.biosafety_threshold,
                                         args.any_safety_threshold)

    # Filter the results
    filter_results(args.input_csv,
                   args.output_csv,
                   valid_topics=valid_topics,
                   min_distance=args.min_distance,
                   max_distance=args.max_distance,
                   remove_errors=not args.keep_errors,
                   filter_by_original_topic=args.filter_by_original_topic)


if __name__ == "__main__":
    main()
