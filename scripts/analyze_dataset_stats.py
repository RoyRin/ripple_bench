#!/usr/bin/env python3
"""
Analyze ripple bench dataset statistics.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter


def analyze_dataset(dataset_path: str):
    """Analyze and print statistics about the ripple bench dataset."""

    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    # Initialize counters
    total_topics = 0
    total_questions = 0
    unique_original_topics = set()
    unique_assigned_question_ids = set()
    distance_distribution = Counter()
    questions_per_topic = []
    original_topic_counts = Counter()

    # Track distance ranges
    distance_ranges = {
        '0-10': 0,
        '11-50': 0,
        '51-100': 0,
        '101-500': 0,
        '501-1000': 0,
        '1000+': 0
    }

    # Analyze topics structure
    if 'topics' in dataset:
        total_topics = len(dataset['topics'])

        for topic_data in dataset['topics']:
            topic_name = topic_data.get('topic', '')

            if 'questions' in topic_data:
                num_questions = len(topic_data['questions'])
                questions_per_topic.append(num_questions)
                total_questions += num_questions

                for q in topic_data['questions']:
                    # Track original topics
                    if 'original_topic' in q:
                        orig_topic = q['original_topic']
                        unique_original_topics.add(orig_topic)
                        original_topic_counts[orig_topic] += 1

                    # Track assigned question IDs
                    if 'assigned_question_id' in q:
                        unique_assigned_question_ids.add(
                            q['assigned_question_id'])

                    # Track distance distribution
                    if 'distance' in q:
                        distance = q['distance']
                        distance_distribution[distance] += 1

                        # Categorize into ranges
                        if distance <= 10:
                            distance_ranges['0-10'] += 1
                        elif distance <= 50:
                            distance_ranges['11-50'] += 1
                        elif distance <= 100:
                            distance_ranges['51-100'] += 1
                        elif distance <= 500:
                            distance_ranges['101-500'] += 1
                        elif distance <= 1000:
                            distance_ranges['501-1000'] += 1
                        else:
                            distance_ranges['1000+'] += 1

    # Print statistics
    print("\n" + "=" * 60)
    print("RIPPLE BENCH DATASET STATISTICS")
    print("=" * 60)

    print("\n📊 Overall Statistics:")
    print(f"  Total topics: {total_topics:,}")
    print(f"  Total questions: {total_questions:,}")
    print(f"  Unique original WMDP topics: {len(unique_original_topics):,}")
    print(
        f"  Unique assigned question IDs: {len(unique_assigned_question_ids):,}"
    )

    if questions_per_topic:
        print(f"\n📈 Questions per topic:")
        print(
            f"  Mean: {sum(questions_per_topic)/len(questions_per_topic):.1f}")
        print(f"  Min: {min(questions_per_topic):,}")
        print(f"  Max: {max(questions_per_topic):,}")
        print(
            f"  Median: {sorted(questions_per_topic)[len(questions_per_topic)//2]:,}"
        )

    print(f"\n🎯 Distance Distribution:")
    if distance_distribution:
        sorted_distances = sorted(distance_distribution.items())
        print(f"  Unique distances: {len(sorted_distances)}")
        print(f"  Min distance: {sorted_distances[0][0]}")
        print(f"  Max distance: {sorted_distances[-1][0]}")

        print(f"\n  Distance Ranges:")
        for range_name, count in sorted(
                distance_ranges.items(),
                key=lambda x: [
                    '0-10', '11-50', '51-100', '101-500', '501-1000', '1000+'
                ].index(x[0])):
            percentage = (count / total_questions *
                          100) if total_questions > 0 else 0
            print(
                f"    {range_name:>10}: {count:>7,} questions ({percentage:>5.1f}%)"
            )

        # Top 10 most common distances
        print(f"\n  Top 10 most common distances:")
        for distance, count in distance_distribution.most_common(10):
            percentage = (count / total_questions *
                          100) if total_questions > 0 else 0
            print(
                f"    Distance {distance:>4}: {count:>6,} questions ({percentage:>5.1f}%)"
            )

    print(f"\n🔬 Original WMDP Topics:")
    print(f"  Total unique original topics: {len(unique_original_topics):,}")
    print(
        f"  Average questions per original topic: {total_questions/len(unique_original_topics):.1f}"
    )

    # Top original topics by question count
    print(f"\n  Top 10 original topics by question count:")
    for orig_topic, count in original_topic_counts.most_common(10):
        print(f"    {orig_topic[:50]:50} {count:>6,} questions")

    # Bottom 10 original topics by question count
    print(f"\n  Bottom 10 original topics by question count:")
    for orig_topic, count in original_topic_counts.most_common()[-10:]:
        print(f"    {orig_topic[:50]:50} {count:>6,} questions")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    dataset_path = "/Users/roy/data/ripple_bench/9_25_2025/data/ripple_bench_2025-bio-9-24/ripple_bench_dataset.json"

    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]

    analyze_dataset(dataset_path)
