#!/usr/bin/env python3
"""
Analyze WMDP questions and show distance-1 topics from Ripple Bench.
Fixed version that properly understands the dataset structure.
"""

import json
from pathlib import Path
import random
from collections import defaultdict


def load_ripple_bench_data(dataset_path):
    """Load Ripple Bench dataset."""
    with open(dataset_path, 'r') as f:
        return json.load(f)


def get_topics_by_distance(ripple_data):
    """Organize topics by their distance."""
    topics_by_distance = defaultdict(list)

    for topic_data in ripple_data['topics']:
        distance = topic_data['distance']
        topics_by_distance[distance].append(topic_data)

    return topics_by_distance


def print_analysis():
    """Main analysis function."""

    # Path to BIO Ripple Bench dataset
    bio_dataset_path = Path(
        '/Users/roy/data/ripple_bench/9_05_2025/data/ripple_bench_2025-09-12-bio/ripple_bench_dataset.json'
    )

    if not bio_dataset_path.exists():
        print(f"Error: Dataset not found at {bio_dataset_path}")
        return

    print("Loading BIO Ripple Bench dataset...")
    ripple_data = load_ripple_bench_data(bio_dataset_path)

    # Organize topics by distance
    topics_by_distance = get_topics_by_distance(ripple_data)

    print(f"\nDataset Overview:")
    print(f"  Total topics: {len(ripple_data['topics'])}")
    print(f"  Distance 0 (WMDP) topics: {len(topics_by_distance[0])}")
    print(f"  Distance 1 topics: {len(topics_by_distance[1])}")
    print(f"  Distance 2 topics: {len(topics_by_distance[2])}")

    # Get all WMDP topics (distance 0)
    wmdp_topics = topics_by_distance[0]
    distance_1_topics = topics_by_distance[1]

    print(f"\n{'='*80}")
    print("UNDERSTANDING RIPPLE BENCH STRUCTURE")
    print(f"{'='*80}")

    print(
        "\nRipple Bench measures semantic distance from dangerous knowledge:")
    print("  - Distance 0: WMDP dangerous topics (e.g., bioweapons, toxins)")
    print("  - Distance 1: Closely related but less dangerous topics")
    print("  - Distance 2+: Progressively less related topics")
    print(
        "\nTopics at each distance are INDEPENDENT - not parent-child relationships."
    )
    print(
        "The 'ripple effect' is that unlearning distance-0 topics may affect")
    print("performance on semantically similar topics at higher distances.")

    # Select 5 random WMDP topics to analyze
    selected_wmdp = random.sample(wmdp_topics, min(5, len(wmdp_topics)))

    print(f"\n{'='*80}")
    print("SAMPLE ANALYSIS: 5 WMDP Topics and Sample Distance-1 Topics")
    print(f"{'='*80}")

    for i, wmdp_topic in enumerate(selected_wmdp, 1):
        print(f"\n{'-'*40}")
        print(f"WMDP Topic #{i} (Distance 0)")
        print(f"{'-'*40}")
        print(f"Topic: {wmdp_topic['topic']}")

        # Show first question from this WMDP topic
        if wmdp_topic.get('questions'):
            q = wmdp_topic['questions'][0]
            print(f"\nSample Question:")
            print(f"  Q: {q['question']}")
            print(f"  A: {q['answer']}")

    # Now show some distance-1 topics (these are NOT directly related to specific WMDP topics)
    print(f"\n{'='*80}")
    print(
        "SAMPLE DISTANCE-1 TOPICS (Semantically closer to dangerous knowledge)"
    )
    print(f"{'='*80}")

    # Select 5 random distance-1 topics
    selected_d1 = random.sample(distance_1_topics,
                                min(5, len(distance_1_topics)))

    for i, d1_topic in enumerate(selected_d1, 1):
        print(f"\n{'-'*40}")
        print(f"Distance-1 Topic #{i}")
        print(f"{'-'*40}")
        print(f"Topic: {d1_topic['topic']}")
        if 'original_topic' in d1_topic:
            print(f"Related concept: {d1_topic['original_topic']}")

        # Show first question
        if d1_topic.get('questions'):
            q = d1_topic['questions'][0]
            print(f"\nSample Question:")
            print(f"  Q: {q['question']}")
            print(f"  A: {q['answer']}")

    print(f"\n{'='*80}")
    print("KEY INSIGHT:")
    print(f"{'='*80}")
    print("Distance-1 topics are NOT children of specific WMDP topics.")
    print("They are independently selected topics that are semantically")
    print("closer to dangerous knowledge than general knowledge topics.")
    print("\nThe 'ripple effect' occurs because unlearning methods that")
    print("forget WMDP topics may also affect performance on these")
    print("semantically related but less dangerous topics.")


def main():
    """Main function."""
    print_analysis()


if __name__ == '__main__':
    main()
