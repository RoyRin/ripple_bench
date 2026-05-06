#!/usr/bin/env python3
"""
Experiment 4: WMDP Question → Topic Extraction Validation
Generates 250 HITs where workers match a WMDP question to its extracted topic.

Sampling strategy:
- Sample 250 WMDP questions
- For each, correct answer = pipeline-extracted topic
- 3 distractors: other extracted topics at moderate embedding similarity (0.3-0.6)
  Approximated by sampling topics 50-200 positions away in alphabetical order
  (since we don't have pre-computed pairwise similarities, this provides
   same-domain but distinct topics). If embeddings are available, use cosine sim.
- Randomize option order
"""

import json
import csv
import random
import argparse
from pathlib import Path

random.seed(42)

DEFAULT_NUM_HITS = 250


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/ripple_bench_2025-bio-9-24/ripple_bench_dataset.json")
    parser.add_argument("--output", default="MTURK/exp4_topic_extraction/exp4_hits.csv")
    parser.add_argument("--num-hits", type=int, default=DEFAULT_NUM_HITS)
    args = parser.parse_args()
    NUM_HITS = args.num_hits

    data = json.load(open(args.dataset))
    topics_df = data['raw_data']['topics_df']  # WMDP questions with extracted topics
    topic_to_neighbors = data['raw_data']['topic_to_neighbors']

    # All unique extracted topics
    all_topics = sorted(set(t['topic'] for t in topics_df))
    topic_to_idx = {t: i for i, t in enumerate(all_topics)}

    # Sample 250 WMDP questions
    eligible = [q for q in topics_df if q['topic'] in topic_to_idx]
    random.shuffle(eligible)
    sampled = eligible[:NUM_HITS]

    hits = []
    for hit_id, q in enumerate(sampled):
        correct_topic = q['topic']
        correct_idx = topic_to_idx[correct_topic]

        # Select 3 distractors: topics that are in the same broad domain
        # but not too close. Use neighbor lists if available, else alphabetical distance.
        distractor_pool = []

        # Strategy: if the correct topic has a neighbor list, pick topics that
        # are neighbors of *other* base topics (related domain, different topic)
        # Fallback: alphabetically distant topics
        for offset in list(range(30, 200)) + list(range(-200, -30)):
            idx = (correct_idx + offset) % len(all_topics)
            candidate = all_topics[idx]
            if candidate != correct_topic:
                distractor_pool.append(candidate)

        if len(distractor_pool) < 3:
            continue

        distractors = random.sample(distractor_pool[:100], 3)

        # Build options in random order
        options = [correct_topic] + distractors
        random.shuffle(options)
        correct_pos = options.index(correct_topic) + 1

        hits.append({
            'hit_id': hit_id,
            'wmdp_question_id': q.get('original_index', hit_id),
            'wmdp_question_text': q['question'],
            'option_1': options[0],
            'option_2': options[1],
            'option_3': options[2],
            'option_4': options[3],
            'correct_option_position': correct_pos,
            'extracted_topic': correct_topic,
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


if __name__ == "__main__":
    main()
