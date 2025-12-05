#!/usr/bin/env python3
"""
Simple script to filter out refusal-related topics from Ripple Bench BIO results.
No pandas dependency - uses only standard library and CSV module.
"""

import json
import csv
from pathlib import Path
import re
from typing import List, Set, Dict, Tuple
import shutil
from datetime import datetime

# Paths
ripple_bench_bio_path = Path(
    "/Users/roy/data/ripple_bench/9_05_2025/data/ripple_bench_2025-09-12-bio")
bio_results_path = Path(
    "/Users/roy/data/ripple_bench/9_05_2025/results/all_models__duplicated__BIO_9_12"
)


def identify_refusal_topics(
        dataset_path: Path) -> Tuple[Set[str], Dict[str, List[str]]]:
    """
    Identify topics that are actually refusals from the model.
    Returns both the refusal topics and their associated question IDs.
    """
    refusal_topics = set()
    refusal_question_ids = {}  # topic -> list of question IDs

    refusal_patterns = [
        r"I can't and won't", r"I cannot and will not",
        r"I'd be happy to discuss", r"potentially be used for harmful",
        r"could facilitate harmful", r"If you're interested in legitimate",
        r"I won't provide information", r"I cannot provide information",
        r"bioweapons", r"dangerous biological agents",
        r"Article not found in local Wikipedia", r" 404"
    ]

    # Load the dataset
    dataset_file = dataset_path / "ripple_bench_dataset.json"
    if not dataset_file.exists():
        print(f"Dataset not found at {dataset_file}")
        return refusal_topics, refusal_question_ids

    with open(dataset_file, 'r') as f:
        data = json.load(f)

    # Check each topic for refusal patterns
    for topic_data in data['topics']:
        topic = topic_data['topic']

        # Check if the topic text matches any refusal pattern
        is_refusal = False
        for pattern in refusal_patterns:
            if re.search(pattern, topic, re.IGNORECASE):
                is_refusal = True
                break

        if is_refusal:
            refusal_topics.add(topic)
            refusal_question_ids[topic] = []

            # Collect question IDs for this refusal topic
            for question in topic_data.get('questions', []):
                # Extract question text to use as identifier
                q_text = question.get('question', '')
                refusal_question_ids[topic].append(q_text)

    return refusal_topics, refusal_question_ids


def filter_csv_file(csv_path: Path, refusal_topics: Set[str],
                    refusal_questions: List[str], output_path: Path) -> int:
    """Filter out rows with refusal topics from CSV results."""
    rows_removed = 0

    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames

            if not fieldnames:
                print(f"Warning: No fieldnames found in {csv_path}")
                shutil.copy2(csv_path, output_path)
                return 0

            with open(output_path, 'w', newline='',
                      encoding='utf-8') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()

                for row in reader:
                    # Check if this row should be filtered
                    should_filter = False

                    # Check topic columns
                    for col in ['topic', 'question_topic']:
                        if col in row and row[col] in refusal_topics:
                            should_filter = True
                            break

                    # Check question column
                    if not should_filter and 'question' in row:
                        question_lower = row['question'].lower()

                        # Check for any problematic patterns
                        if ("Article not found in local Wikipedia"
                                in row['question']
                                or "wikipedia" in question_lower
                                or row['question'] in refusal_questions
                                or any(phrase in question_lower for phrase in [
                                    "i can't", "i cannot", "i won't",
                                    "harmful", "bioweapon",
                                    "article not found", " 404"
                                ])):
                            should_filter = True

                    # Also check choices column if it exists (answers may contain "wikipedia")
                    if not should_filter and 'choices' in row:
                        if "wikipedia" in row['choices'].lower():
                            should_filter = True

                    if not should_filter:
                        writer.writerow(row)
                    else:
                        rows_removed += 1

        return rows_removed

    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        # If error, just copy the original file
        shutil.copy2(csv_path, output_path)
        return 0


def main():
    """Main function to filter refusal topics from results."""

    print("=" * 80)
    print("RIPPLE BENCH REFUSAL FILTERING (Simple Version)")
    print("=" * 80)

    # Step 1: Identify refusal topics
    print("\n1. Identifying refusal topics...")
    refusal_topics, refusal_question_ids = identify_refusal_topics(
        ripple_bench_bio_path)

    # Flatten all refusal questions into a single list
    all_refusal_questions = []
    for questions in refusal_question_ids.values():
        all_refusal_questions.extend(questions)

    print(f"   Found {len(refusal_topics)} refusal topics")
    print(f"   Found {len(all_refusal_questions)} associated questions")

    if refusal_topics:
        print("\n   Refusal topics found:")
        for i, topic in enumerate(refusal_topics, 1):
            # Truncate long topics for display
            display_topic = topic[:100] + "..." if len(topic) > 100 else topic
            print(f"     {i}. {display_topic}")

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
    processed_csv = 0
    processed_json = 0

    # Process CSV files
    print("\n   Processing CSV files...")
    for csv_file in csv_files:
        output_file = output_dir / csv_file.name
        rows_removed = filter_csv_file(csv_file, refusal_topics,
                                       all_refusal_questions, output_file)
        total_rows_removed += rows_removed
        processed_csv += 1

        if rows_removed > 0:
            print(f"     {csv_file.name}: removed {rows_removed} rows")

    # Copy JSON files with a note about filtering
    print("\n   Processing JSON summary files...")
    for json_file in json_files:
        output_file = output_dir / json_file.name
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Add filtering metadata
            data['filtered'] = True
            data['refusal_topics_removed'] = len(refusal_topics)
            data['filter_timestamp'] = timestamp

            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)

            processed_json += 1

        except Exception as e:
            print(f"     Warning: Could not process {json_file.name}: {e}")
            shutil.copy2(json_file, output_file)

    print(f"\n4. Summary:")
    print(f"   - Processed {processed_csv} CSV files")
    print(f"   - Processed {processed_json} JSON files")
    print(
        f"   - Removed {total_rows_removed} total rows containing refusal topics/questions"
    )
    print(f"   - Filtered results saved to:\n     {output_dir}")

    # Step 4: Save refusal topics list for reference
    refusal_list_path = output_dir / "removed_refusal_topics.json"
    with open(refusal_list_path, 'w') as f:
        json.dump(
            {
                'refusal_topics': list(refusal_topics),
                'refusal_questions_sample':
                all_refusal_questions[:10],  # Save sample of questions
                'total_refusal_questions': len(all_refusal_questions),
                'count': len(refusal_topics),
                'timestamp': timestamp
            },
            f,
            indent=2)

    print(f"   - List of removed topics saved to: removed_refusal_topics.json")

    print("\n" + "=" * 80)
    print("Filtering complete!")
    print(f"You can now use the filtered results from:\n{output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
