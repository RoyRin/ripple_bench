#!/usr/bin/env python3
"""
Batch filter all evaluation results in a directory.
Applies the same filtering to all CSV files and saves them with a suffix.
"""

import pandas as pd
import json
from pathlib import Path
import argparse
from typing import Set, Optional
import sys

# Import the filtering functions from the single-file script
sys.path.append(str(Path(__file__).parent))
from filter_results_new_format import load_valid_topics, filter_results


def process_directory(input_dir: str,
                      output_dir: str,
                      suffix: str = "_filtered",
                      valid_topics: Optional[Set[str]] = None,
                      min_distance: Optional[int] = None,
                      max_distance: Optional[int] = None,
                      remove_errors: bool = True,
                      filter_by_original_topic: bool = False):
    """
    Process all CSV files in a directory.

    Args:
        input_dir: Directory containing CSV files to filter
        output_dir: Directory to save filtered CSV files
        suffix: Suffix to add to filtered files (before .csv)
        valid_topics: Set of valid topics to keep
        min_distance: Minimum distance to include
        max_distance: Maximum distance to include
        remove_errors: Whether to remove ERROR responses
        filter_by_original_topic: Filter by original_topic instead of topic
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

    # Process each file
    for csv_file in sorted(csv_files):
        # Create output filename
        output_name = csv_file.stem + suffix + ".csv"
        output_file = output_path / output_name

        print(f"\nProcessing: {csv_file.name}")

        try:
            filter_results(str(csv_file),
                           str(output_file),
                           valid_topics=valid_topics,
                           min_distance=min_distance,
                           max_distance=max_distance,
                           remove_errors=remove_errors,
                           filter_by_original_topic=filter_by_original_topic)
        except Exception as e:
            print(f"  ERROR: Failed to process {csv_file.name}: {e}")
            continue

    print("\n" + "=" * 60)
    print(f"Filtering complete! Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch filter evaluation results")

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

    # Process all files in directory
    process_directory(args.input_dir,
                      args.output_dir,
                      suffix=args.suffix,
                      valid_topics=valid_topics,
                      min_distance=args.min_distance,
                      max_distance=args.max_distance,
                      remove_errors=not args.keep_errors,
                      filter_by_original_topic=args.filter_by_original_topic)


if __name__ == "__main__":
    main()
