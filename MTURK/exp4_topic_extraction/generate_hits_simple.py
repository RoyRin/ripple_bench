#!/usr/bin/env python3
"""
Experiment 4 (simple variant): Question → Base Topic Extraction
Shows a generated question and asks workers to identify which of 4 base topics it belongs to.
Validates: does the question clearly relate to its base topic?

Each of the 30 base topics gets one HIT, using a question generated at distance 0
(i.e., a question about the base topic itself).
"""

import json
import csv
import random
import argparse
from pathlib import Path

random.seed(42)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/ripple_bench_human_eval_100/ripple_bench_dataset.json")
    parser.add_argument("--output", default="MTURK/exp4_topic_extraction/exp4_hits_simple.csv")
    args = parser.parse_args()

    data = json.load(open(args.dataset))

    # Get the 30 base topics
    base_topics = sorted(data['raw_data']['topic_to_neighbors'].keys())

    # For each base topic, find a question at the lowest available distance
    topic_to_question = {}
    for dist in [0, 1, 5, 10]:  # prefer distance 0, fallback to close neighbors
        for topic_entry in data['topics']:
            if topic_entry['distance'] == dist and topic_entry['original_topic'] in base_topics:
                if topic_entry['original_topic'] not in topic_to_question:
                    if topic_entry['questions']:
                        topic_to_question[topic_entry['original_topic']] = topic_entry['questions'][0]

    hits = []
    for base_topic in base_topics:
        if base_topic not in topic_to_question:
            continue

        q = topic_to_question[base_topic]

        # Pick 3 distractor base topics
        other_topics = [t for t in base_topics if t != base_topic]
        distractors = random.sample(other_topics, 3)

        options = [base_topic] + distractors
        random.shuffle(options)
        correct_pos = options.index(base_topic) + 1

        hits.append({
            'hit_id': len(hits),
            'wmdp_question_id': q.get('assigned_question_id', len(hits)),
            'wmdp_question_text': q['question'],
            'option_1': options[0],
            'option_2': options[1],
            'option_3': options[2],
            'option_4': options[3],
            'correct_option_position': correct_pos,
            'extracted_topic': base_topic,
            'distractor_1': distractors[0],
            'distractor_2': distractors[1],
            'distractor_3': distractors[2],
        })

    random.shuffle(hits)
    for i, h in enumerate(hits):
        h['hit_id'] = i

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=hits[0].keys())
        writer.writeheader()
        writer.writerows(hits)

    print(f"Generated {len(hits)} HITs -> {output}")

    # Show a few examples
    for h in hits[:3]:
        print(f"  Q: {h['wmdp_question_text'][:80]}...")
        print(f"  Correct: {h['extracted_topic']}")
        print(f"  Options: {h['option_1']}, {h['option_2']}, {h['option_3']}, {h['option_4']}")
        print()


if __name__ == "__main__":
    main()
