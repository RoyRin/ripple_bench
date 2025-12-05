#!/usr/bin/env python3
"""
Analyze CHEM or BIO ripple bench topics and create a comprehensive list.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

# Data paths
wmdp_bio_path = Path(
    "/Users/roy/data/ripple_bench/9_05_2025/data/wmdp/wmdp-bio.json")
ripple_bench_bio_path = Path(
    "/Users/roy/data/ripple_bench/9_05_2025/data/ripple_bench_2025-09-12-bio/ripple_bench_dataset.json"
)
bio_results_path = Path(
    "/Users/roy/data/ripple_bench/9_05_2025/results/all_models__duplicated__BIO"
)

wmdp_chem_path = Path(
    "/Users/roy/data/ripple_bench/9_05_2025/data/wmdp/wmdp-chem.json")
ripple_bench_chem_path = Path(
    "/Users/roy/data/ripple_bench/9_05_2025/data/ripple_bench_2025-09-12-chem/ripple_bench_dataset.json"
)
chem_results_path = Path(
    "/Users/roy/data/ripple_bench/9_05_2025/results/all_models__duplicated__CHEM"
)


def analyze_topics(dataset_path, dataset_type, output_file=None, verbose=True):
    """Create a comprehensive list of all topics organized by distance."""

    if output_file is None:
        output_file = f'{dataset_type.lower()}_topics_by_distance.txt'

    # Load ripple bench dataset
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    # Organize topics by distance
    topics_by_distance = defaultdict(list)
    for topic_data in data['topics']:
        dist = topic_data['distance']
        topic = topic_data['topic']
        num_questions = len(topic_data.get('questions', []))
        topics_by_distance[dist].append({
            'topic': topic,
            'num_questions': num_questions
        })

    # Write to output file
    with open(output_file, 'w') as f:
        f.write(
            f'{dataset_type.upper()} Ripple Bench - Complete Topic List by Distance\n'
        )
        f.write('=' * 70 + '\n\n')

        # Summary statistics
        all_distances = sorted(topics_by_distance.keys())
        total_topics = sum(
            len(topics) for topics in topics_by_distance.values())
        f.write(f'Total Topics: {total_topics}\n')
        f.write(
            f'Distance Range: {min(all_distances)} to {max(all_distances)}\n')
        f.write(f'Number of Unique Distances: {len(all_distances)}\n\n')

        # Topics by distance
        for dist in all_distances:
            topics = topics_by_distance[dist]
            f.write(f'\n--- Distance {dist} ({len(topics)} topics) ---\n')
            for topic_info in sorted(topics, key=lambda x: x['topic']):
                f.write(
                    f"  {topic_info['topic']} ({topic_info['num_questions']} questions)\n"
                )

        # Distance range summary
        f.write('\n\n' + '=' * 70 + '\n')
        f.write('Summary by Distance Ranges:\n')
        f.write('-' * 30 + '\n')

        ranges = [(0, 10, "Very Close (0-10)"), (11, 50, "Close (11-50)"),
                  (51, 100, "Medium (51-100)"), (101, 200, "Far (101-200)"),
                  (201, 500, "Very Far (201-500)"),
                  (501, 1000, "Extremely Far (501-1000)")]

        for low, high, label in ranges:
            count = sum(
                len(topics_by_distance[d]) for d in range(low, high + 1)
                if d in topics_by_distance)
            f.write(f'{label}: {count} topics\n')

    if verbose:
        print(f'Topic analysis saved to: {output_file}')

        # Also print a summary
        print('\nQuick Summary:')
        print('-' * 40)
        for low, high, label in ranges:
            count = sum(
                len(topics_by_distance[d]) for d in range(low, high + 1)
                if d in topics_by_distance)
            example_topics = []
            for d in range(low, min(high + 1, max(all_distances) + 1)):
                if d in topics_by_distance and example_topics.__len__() < 3:
                    example_topics.extend([
                        t['topic']
                        for t in topics_by_distance[d][:3 -
                                                       len(example_topics)]
                    ])

            print(f'\n{label}: {count} topics')
            if example_topics:
                print(f'  Examples: {", ".join(example_topics[:3])}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze CHEM or BIO ripple bench topics')
    parser.add_argument('dataset',
                        choices=['chem', 'bio'],
                        help='Dataset to analyze: chem or bio')
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        help='Output file name (default: {dataset}_topics_by_distance.txt)')
    parser.add_argument('--quiet',
                        '-q',
                        action='store_true',
                        help='Suppress verbose output (default: verbose=True)')

    args = parser.parse_args()

    # Select paths based on dataset type
    if args.dataset == 'chem':
        dataset_path = ripple_bench_chem_path
        dataset_type = 'CHEM'
    else:  # bio
        dataset_path = ripple_bench_bio_path
        dataset_type = 'BIO'

    # Check if dataset file exists
    if not dataset_path.exists():
        print(f"Error: Dataset file not found: {dataset_path}")
        exit(1)

    verbose = not args.quiet
    if verbose:
        print(f"Analyzing {dataset_type} dataset from: {dataset_path}")

    # Run analysis
    analyze_topics(dataset_path, dataset_type, args.output, verbose)
