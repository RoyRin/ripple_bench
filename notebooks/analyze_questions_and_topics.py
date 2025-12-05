#!/usr/bin/env python3
"""
Analyze WMDP questions and their corresponding distance-1 topics in Ripple Bench.
"""

import json
from pathlib import Path
import random


def load_ripple_bench_data(dataset_path):
    """Load Ripple Bench dataset."""
    with open(dataset_path, 'r') as f:
        return json.load(f)


def get_wmdp_questions(ripple_data):
    """Extract WMDP questions (distance=0) from Ripple Bench data."""
    wmdp_questions = []
    for topic_data in ripple_data['topics']:
        if topic_data['distance'] == 0:
            for question in topic_data.get('questions', []):
                wmdp_questions.append({
                    'topic': topic_data['topic'],
                    'question': question['question'],
                    'answer': question['answer'],
                    'choices': question.get('choices', [])
                })
    return wmdp_questions


def get_distance_1_questions_for_topic(ripple_data, wmdp_topic):
    """Find distance-1 questions for any topic."""
    distance_1_topics = []

    # Simply collect all distance-1 topics
    all_dist1_topics = []
    for topic_data in ripple_data['topics']:
        if topic_data['distance'] == 1:
            all_dist1_topics.append(topic_data)

    # Randomly select a few distance-1 topics to show
    import random
    if all_dist1_topics:
        selected = random.sample(all_dist1_topics, min(3,
                                                       len(all_dist1_topics)))
        for topic_data in selected:
            distance_1_topics.append({
                'topic':
                topic_data['topic'],
                'questions':
                topic_data.get('questions', [])[:5]
            })

    return distance_1_topics


def print_question_analysis(wmdp_question, distance_1_data, question_num):
    """Pretty print the analysis for one WMDP question and its distance-1 counterparts."""
    print(f"\n{'='*80}")
    print(f"ANALYSIS #{question_num}")
    print(f"{'='*80}")

    print(f"\n[WMDP Question - Distance 0]")
    print(f"Topic: {wmdp_question['topic']}")
    print(f"Question: {wmdp_question['question']}")
    print(f"Answer: {wmdp_question['answer']}")
    if wmdp_question['choices']:
        print(f"Choices:")
        for i, choice in enumerate(wmdp_question['choices']):
            print(f"  {chr(65+i)}) {choice}")

    print(f"\n[Related Distance-1 Topics and Questions]")

    if not distance_1_data:
        print("  No distance-1 topics found!")
    else:
        print(f"  Found {len(distance_1_data)} distance-1 topics")

    for dist1_topic_data in distance_1_data[:
                                            2]:  # Show max 2 distance-1 topics
        print(f"\nDistance-1 Topic: {dist1_topic_data['topic']}")
        print(
            f"Number of questions available: {len(dist1_topic_data['questions'])}"
        )

        if not dist1_topic_data['questions']:
            print("  No questions available for this topic")
        else:
            # Show only 2 questions per distance-1 topic
            for i, question in enumerate(dist1_topic_data['questions'][:2], 1):
                print(f"\n  Question {i}:")
                print(f"    Q: {question['question']}")
                print(f"    A: {question['answer']}")
                if question.get('choices'):
                    print(f"    Choices:")
                    for j, choice in enumerate(question['choices']):
                        print(f"      {chr(65+j)}) {choice}")


def main():
    """Main function to analyze WMDP questions and their distance-1 counterparts."""

    # Path to BIO Ripple Bench dataset
    bio_dataset_path = Path(
        '/Users/roy/data/ripple_bench/9_05_2025/data/ripple_bench_2025-09-12-bio/ripple_bench_dataset.json'
    )

    if not bio_dataset_path.exists():
        print(f"Error: Dataset not found at {bio_dataset_path}")
        print("Please check the path to the BIO Ripple Bench dataset.")
        return

    print("Loading BIO Ripple Bench dataset...")
    ripple_data = load_ripple_bench_data(bio_dataset_path)

    # Count topics by distance to understand the data
    distance_counts = {}
    for topic_data in ripple_data['topics']:
        dist = topic_data['distance']
        distance_counts[dist] = distance_counts.get(dist, 0) + 1

    print(f"\nDataset statistics:")
    print(f"  Total topics: {len(ripple_data['topics'])}")
    print(f"  Topics at distance 0 (WMDP): {distance_counts.get(0, 0)}")
    print(f"  Topics at distance 1: {distance_counts.get(1, 0)}")
    print(f"  Topics at distance 2: {distance_counts.get(2, 0)}")

    # Get WMDP questions (distance=0)
    wmdp_questions = get_wmdp_questions(ripple_data)
    print(f"\nFound {len(wmdp_questions)} WMDP questions (distance=0)")

    if not wmdp_questions:
        print("No WMDP questions found in the dataset.")
        return

    # Randomly select 5 WMDP questions
    selected_questions = random.sample(wmdp_questions,
                                       min(5, len(wmdp_questions)))

    print(
        f"\nAnalyzing {len(selected_questions)} randomly selected WMDP questions..."
    )

    # Analyze each selected question
    for i, wmdp_question in enumerate(selected_questions, 1):
        # Find related distance-1 questions
        distance_1_data = get_distance_1_questions_for_topic(
            ripple_data, wmdp_question['topic'])

        # Print the analysis
        print_question_analysis(wmdp_question, distance_1_data, i)

    print(f"\n{'='*80}")
    print("Analysis complete!")


if __name__ == '__main__':
    main()
