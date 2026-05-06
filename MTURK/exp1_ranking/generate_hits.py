#!/usr/bin/env python3
"""
Experiment 1: Ranking Validation (Pairwise Comparison)
Generates 250 HITs for pairwise topic-relatedness judgments.

Sampling strategy:
- Select base topics randomly from WMDP-extracted topics
- For each HIT, sample a stratified pair: one topic from a lower-rank bucket,
  one from a higher-rank bucket
- Bucket boundaries: [1-10], [10-50], [50-100], [100-250], [250-500], [500-1000]
- Spread bucket-pair combinations across 250 HITs
- Randomize display order of Topic A / Topic B
"""

import json
import csv
import random
import argparse
from pathlib import Path
from itertools import combinations

random.seed(42)

BUCKETS = [
    (1, 10),
    (10, 50),
    (50, 100),
    (100, 250),
    (250, 500),
    (500, 1000),
]

DEFAULT_NUM_HITS = 250


def sample_from_bucket(neighbors, bucket_lo, bucket_hi):
    """Sample a random topic from neighbors in the given rank range."""
    candidates = []
    for rank, topic in enumerate(neighbors):
        if bucket_lo <= rank < bucket_hi:
            candidates.append((rank, topic))
    if not candidates:
        return None
    return random.choice(candidates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/ripple_bench_2025-bio-9-24/ripple_bench_dataset.json")
    parser.add_argument("--output", default="MTURK/exp1_ranking/exp1_hits.csv")
    parser.add_argument("--num-hits", type=int, default=DEFAULT_NUM_HITS)
    args = parser.parse_args()
    NUM_HITS = args.num_hits

    data = json.load(open(args.dataset))
    topic_to_neighbors = data['raw_data']['topic_to_neighbors']

    # All bucket pairs (lower bucket index, higher bucket index)
    bucket_pairs = list(combinations(range(len(BUCKETS)), 2))
    # How many HITs per bucket pair
    hits_per_pair = NUM_HITS // len(bucket_pairs)  # 250 // 15 = 16
    remainder = NUM_HITS % len(bucket_pairs)       # 250 % 15 = 10

    base_topics = list(topic_to_neighbors.keys())
    random.shuffle(base_topics)

    hits = []
    hit_id = 0

    for pair_idx, (bi, bj) in enumerate(bucket_pairs):
        bucket_lo = BUCKETS[bi]
        bucket_hi = BUCKETS[bj]
        n_hits = hits_per_pair + (1 if pair_idx < remainder else 0)

        attempts = 0
        generated = 0
        while generated < n_hits and attempts < n_hits * 20:
            base_topic = random.choice(base_topics)
            neighbors = topic_to_neighbors[base_topic]
            attempts += 1

            sample_close = sample_from_bucket(neighbors, bucket_lo[0], bucket_lo[1])
            sample_far = sample_from_bucket(neighbors, bucket_hi[0], bucket_hi[1])

            if sample_close is None or sample_far is None:
                continue

            rank_close, topic_close = sample_close
            rank_far, topic_far = sample_far

            # Randomize display order
            if random.random() < 0.5:
                topic_a, rank_a = topic_close, rank_close
                topic_b, rank_b = topic_far, rank_far
                closer_shown_as = "A"
            else:
                topic_a, rank_a = topic_far, rank_far
                topic_b, rank_b = topic_close, rank_close
                closer_shown_as = "B"

            bucket_pair_label = f"{bucket_lo[0]}-{bucket_lo[1]}_vs_{bucket_hi[0]}-{bucket_hi[1]}"

            hits.append({
                'hit_id': hit_id,
                'base_topic': base_topic,
                'topic_a': topic_a,
                'topic_a_rank': rank_a,
                'topic_b': topic_b,
                'topic_b_rank': rank_b,
                'bucket_pair': bucket_pair_label,
                'closer_shown_as': closer_shown_as,
            })
            hit_id += 1
            generated += 1

    # Shuffle all HITs
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

    # Print bucket pair distribution
    from collections import Counter
    bp_counts = Counter(h['bucket_pair'] for h in hits)
    for bp, count in sorted(bp_counts.items()):
        print(f"  {bp}: {count}")


if __name__ == "__main__":
    main()
