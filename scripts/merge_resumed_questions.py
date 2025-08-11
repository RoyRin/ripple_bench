#!/usr/bin/env python3
"""
Merge resumed questions back into the main ripple bench dataset.
"""

import json
from pathlib import Path
import argparse
from datetime import datetime
import shutil


def merge_questions(output_dir):
    """Merge resumed questions into the main dataset."""

    output_dir = Path(output_dir)

    # Paths
    questions_dir = output_dir / "intermediate" / "questions"
    resumed_file = questions_dir / "questions_resumed.json"

    # Find the main dataset file
    dataset_files = list(output_dir.glob("ripple_bench_dataset*.json"))
    if not dataset_files:
        print("Error: No ripple_bench_dataset file found!")
        return False

    # Use the most recent one if multiple exist
    dataset_file = sorted(dataset_files)[-1]
    print(f"Found dataset file: {dataset_file}")

    # Check if resumed questions exist
    if not resumed_file.exists():
        print("No resumed questions file found. Nothing to merge.")
        return False

    # Load the main dataset
    print("Loading main dataset...")
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)

    # Load resumed questions
    print("Loading resumed questions...")
    with open(resumed_file, 'r') as f:
        resumed_questions = json.load(f)

    print(f"Resumed questions: {len(resumed_questions)}")

    # Create backup
    backup_file = dataset_file.with_suffix('.backup.json')
    print(f"Creating backup at: {backup_file}")
    shutil.copy2(dataset_file, backup_file)

    # Update the questions in raw_data
    if 'raw_data' in dataset and 'questions' in dataset['raw_data']:
        original_count = len(dataset['raw_data']['questions'])
        print(f"Original questions in dataset: {original_count}")

        # Replace with resumed questions (which includes all questions)
        dataset['raw_data']['questions'] = resumed_questions

        # Update metadata
        dataset['metadata']['total_generated_questions'] = len(
            resumed_questions)
        dataset['metadata']['last_updated'] = datetime.now().isoformat()
        dataset['metadata']['resumed'] = True

        print(f"Updated questions: {len(resumed_questions)}")

    else:
        print("Error: Dataset structure not as expected!")
        return False

    # Save updated dataset
    print(f"Saving updated dataset to: {dataset_file}")
    with open(dataset_file, 'w') as f:
        json.dump(dataset, f, indent=2)

    # Also update the topics data if needed
    # Group questions by topic for the topics section
    topics_data = dataset.get('topics', {})
    questions_by_topic = {}

    for q in resumed_questions:
        topic = q.get('topic')
        if topic:
            if topic not in questions_by_topic:
                questions_by_topic[topic] = []
            questions_by_topic[topic].append(q)

    # Update topics section with new questions
    topics_updated = 0
    for topic_entry in topics_data:
        topic_name = topic_entry.get('topic')
        if topic_name in questions_by_topic:
            topic_entry['questions'] = questions_by_topic[topic_name]
            topics_updated += 1

    if topics_updated > 0:
        print(
            f"Updated questions for {topics_updated} topics in the topics section"
        )

        # Save again with updated topics
        with open(dataset_file, 'w') as f:
            json.dump(dataset, f, indent=2)

    print("\n✅ Successfully merged resumed questions into dataset!")
    print(f"Backup saved at: {backup_file}")

    # Optionally remove the resumed file
    print("\nYou can now remove the temporary resumed file:")
    print(f"  rm {resumed_file}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Merge resumed questions into ripple bench dataset")
    parser.add_argument("--output-dir",
                        required=True,
                        help="Ripple bench output directory")

    args = parser.parse_args()

    success = merge_questions(args.output_dir)

    if not success:
        print("\n❌ Failed to merge questions")
        exit(1)


if __name__ == "__main__":
    main()
