#!/usr/bin/env python3
"""
Identify and filter out refusal-related topics from Ripple Bench BIO results.
"""

import json
import pandas as pd
from pathlib import Path
import re
from typing import List, Set, Dict
import shutil
from datetime import datetime

# Paths
ripple_bench_bio_path = Path(
    "/Users/roy/data/ripple_bench/9_05_2025/data/ripple_bench_2025-09-12-bio")
bio_results_path = Path(
    "/Users/roy/data/ripple_bench/9_05_2025/results/all_models__duplicated__BIO_9_12"
)


def identify_refusal_topics(dataset_path: Path) -> Set[str]:
    """Identify topics that are actually refusals from the model."""
    refusal_topics = set()
    refusal_patterns = [
        r"I can't and won't", r"I cannot and will not",
        r"I'd be happy to discuss", r"potentially be used for harmful",
        r"could facilitate harmful", r"If you're interested in legitimate",
        r"I won't provide information", r"I cannot provide information",
        r"bioweapons", r"dangerous biological agents"
    ]

    # Load the dataset
    dataset_file = dataset_path / "ripple_bench_dataset.json"
    if not dataset_file.exists():
        print(f"Dataset not found at {dataset_file}")
        return refusal_topics

    with open(dataset_file, 'r') as f:
        data = json.load(f)

    # Check each topic for refusal patterns
    for topic_data in data['topics']:
        topic = topic_data['topic']

        # Check if the topic text matches any refusal pattern
        for pattern in refusal_patterns:
            if re.search(pattern, topic, re.IGNORECASE):
                refusal_topics.add(topic)
                break

    return refusal_topics


def get_refusal_question_ids(dataset_path: Path,
                             refusal_topics: Set[str]) -> Set[str]:
    """Get all question IDs associated with refusal topics."""
    refusal_question_ids = set()

    dataset_file = dataset_path / "ripple_bench_dataset.json"
    with open(dataset_file, 'r') as f:
        data = json.load(f)

    for topic_data in data['topics']:
        if topic_data['topic'] in refusal_topics:
            for question in topic_data.get('questions', []):
                # Question IDs might be stored as 'id' or need to be constructed
                if 'id' in question:
                    refusal_question_ids.add(question['id'])
                elif 'question_id' in question:
                    refusal_question_ids.add(question['question_id'])
                else:
                    # Create a unique ID from question text hash
                    import hashlib
                    q_text = question.get('question', '')
                    q_id = hashlib.md5(q_text.encode()).hexdigest()[:10]
                    refusal_question_ids.add(q_id)

    return refusal_question_ids


def filter_csv_results(csv_path: Path, refusal_topics: Set[str],
                       output_path: Path) -> int:
    """Filter out rows with refusal topics from CSV results."""
    try:
        df = pd.read_csv(csv_path)
        original_len = len(df)

        # Filter out rows where the topic is a refusal
        if 'topic' in df.columns:
            df_filtered = df[~df['topic'].isin(refusal_topics)]
        elif 'question_topic' in df.columns:
            df_filtered = df[~df['question_topic'].isin(refusal_topics)]
        else:
            # If no topic column, check question text for refusal patterns
            refusal_patterns = [
                r"I can't and won't", r"I cannot and will not",
                r"potentially be used for harmful", r"could facilitate harmful"
            ]

            mask = pd.Series(True, index=df.index)
            if 'question' in df.columns:
                for pattern in refusal_patterns:
                    mask &= ~df['question'].str.contains(
                        pattern, case=False, na=False)
            df_filtered = df[mask]

        filtered_len = len(df_filtered)
        rows_removed = original_len - filtered_len

        # Save filtered results
        df_filtered.to_csv(output_path, index=False)

        return rows_removed

    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return 0


def filter_json_summary(json_path: Path, refusal_topics: Set[str],
                        output_path: Path) -> bool:
    """Filter and recalculate summary JSON."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Note: We're keeping the JSON as-is but could add a note about filtering
        data['filtered'] = True
        data['refusal_topics_removed'] = len(refusal_topics)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        return True

    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return False


def main():
    """Main function to filter refusal topics from results."""

    print("=" * 80)
    print("RIPPLE BENCH REFUSAL FILTERING")
    print("=" * 80)

    # Step 1: Identify refusal topics
    print("\n1. Identifying refusal topics...")
    refusal_topics = identify_refusal_topics(ripple_bench_bio_path)

    print(f"   Found {len(refusal_topics)} refusal topics")

    if refusal_topics:
        print("\n   Sample refusal topics:")
        for topic in list(refusal_topics)[:5]:
            # Truncate long topics for display
            display_topic = topic[:100] + "..." if len(topic) > 100 else topic
            print(f"     - {display_topic}")

    if not refusal_topics:
        print("   No refusal topics found. Nothing to filter.")
        return

    # Step 2: Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = bio_results_path.parent / f"all_models__duplicated__BIO_9_12_filtered_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n2. Creating filtered output directory:\n   {output_dir}")

    # Step 3: Process all result files
    print("\n3. Processing result files...")

    csv_files = list(bio_results_path.glob("*.csv"))
    json_files = list(bio_results_path.glob("*.json"))

    print(
        f"   Found {len(csv_files)} CSV files and {len(json_files)} JSON files"
    )

    total_rows_removed = 0
    processed_files = 0

    # Process CSV files
    for csv_file in csv_files:
        output_file = output_dir / csv_file.name
        rows_removed = filter_csv_results(csv_file, refusal_topics,
                                          output_file)
        total_rows_removed += rows_removed
        processed_files += 1

        if rows_removed > 0:
            print(f"   Filtered {csv_file.name}: removed {rows_removed} rows")

    # Process JSON files
    for json_file in json_files:
        output_file = output_dir / json_file.name
        if filter_json_summary(json_file, refusal_topics, output_file):
            processed_files += 1

    print(f"\n4. Summary:")
    print(f"   - Processed {processed_files} files")
    print(
        f"   - Removed {total_rows_removed} total rows containing refusal topics"
    )
    print(f"   - Filtered results saved to: {output_dir}")

    # Step 4: Save refusal topics list for reference
    refusal_list_path = output_dir / "removed_refusal_topics.json"
    with open(refusal_list_path, 'w') as f:
        json.dump(
            {
                'refusal_topics': list(refusal_topics),
                'count': len(refusal_topics),
                'timestamp': timestamp
            },
            f,
            indent=2)

    print(f"   - List of removed topics saved to: removed_refusal_topics.json")

    print("\n" + "=" * 80)
    print("Filtering complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
