#!/usr/bin/env python3
"""
Experiment 2: Question Quality Validation (Fact-Grounded MCQ)
Generates 250 HITs: 125 with facts (Condition A), 125 without (Condition B).

Sampling strategy:
- 250 questions from RippleBench-Bio, stratified across 8 distance buckets
- Randomly assign 125 to Condition A (with facts), 125 to Condition B (without)
- Balanced ~15-16 per bucket per condition
"""

import json
import csv
import random
import argparse
from pathlib import Path

random.seed(42)

DISTANCE_BUCKETS = [
    ("1-5", 1, 5),
    ("6-25", 6, 25),
    ("26-50", 26, 50),
    ("51-100", 51, 100),
    ("101-200", 101, 200),
    ("201-350", 201, 350),
    ("351-600", 351, 600),
    ("601-1000", 601, 1000),
]

DEFAULT_NUM_HITS = 250


def clean_choice(choice):
    """Remove letter prefix like 'A) ' from choices."""
    choice = choice.strip()
    if len(choice) > 2 and choice[0] in 'ABCD' and choice[1] in ').:-':
        return choice[2:].strip()
    return choice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/ripple_bench_2025-bio-9-24/ripple_bench_dataset.json")
    parser.add_argument("--output", default="MTURK/exp2_question_quality/exp2_hits.csv")
    parser.add_argument("--num-hits", type=int, default=DEFAULT_NUM_HITS)
    args = parser.parse_args()
    NUM_HITS = args.num_hits
    PER_BUCKET = NUM_HITS // len(DISTANCE_BUCKETS)

    data = json.load(open(args.dataset))

    # Build a flat list of (question, facts, distance) from topics
    all_questions = []
    for topic_entry in data['topics']:
        distance = topic_entry['distance']
        facts = topic_entry.get('facts', '')
        for q in topic_entry['questions']:
            all_questions.append({
                'question': q['question'],
                'choices': q['choices'],
                'answer': q['answer'],
                'distance': distance,
                'topic': q.get('topic', topic_entry.get('topic', '')),
                'original_topic': q.get('original_topic', topic_entry.get('original_topic', '')),
                'assigned_question_id': q.get('assigned_question_id', ''),
                'facts': facts,
            })

    # Sample per bucket
    sampled = []
    for label, lo, hi in DISTANCE_BUCKETS:
        bucket_qs = [q for q in all_questions if lo <= q['distance'] <= hi]
        random.shuffle(bucket_qs)

        # Deduplicate by assigned_question_id
        seen_ids = set()
        unique_qs = []
        for q in bucket_qs:
            qid = q['assigned_question_id']
            if qid not in seen_ids:
                seen_ids.add(qid)
                unique_qs.append(q)

        n = min(PER_BUCKET + 2, len(unique_qs))  # slight oversample, trim later
        sampled.extend([(q, label) for q in unique_qs[:n]])

    random.shuffle(sampled)
    sampled = sampled[:NUM_HITS]

    # Assign conditions: balanced across buckets
    # Group by bucket, split each bucket in half
    by_bucket = {}
    for q, label in sampled:
        by_bucket.setdefault(label, []).append(q)

    hits = []
    hit_id = 0
    for label in by_bucket:
        qs = by_bucket[label]
        half = len(qs) // 2
        for i, q in enumerate(qs):
            condition = "with_facts" if i < half else "without_facts"
            choices = [clean_choice(c) for c in q['choices']]

            hits.append({
                'hit_id': hit_id,
                'question_id': q['assigned_question_id'],
                'question_text': q['question'],
                'option_a': choices[0] if len(choices) > 0 else '',
                'option_b': choices[1] if len(choices) > 1 else '',
                'option_c': choices[2] if len(choices) > 2 else '',
                'option_d': choices[3] if len(choices) > 3 else '',
                'correct_answer': q['answer'],
                'condition': condition,
                'facts_json': json.dumps(q['facts']) if condition == "with_facts" else '',
                'semantic_distance': q['distance'],
                'distance_bucket': label,
            })
            hit_id += 1

    # Interleave with_facts and without_facts so MTurk doesn't show long runs of one condition
    wf = [h for h in hits if h['condition'] == 'with_facts']
    wo = [h for h in hits if h['condition'] == 'without_facts']
    random.shuffle(wf)
    random.shuffle(wo)
    hits = []
    while wf or wo:
        if wf:
            hits.append(wf.pop())
        if wo:
            hits.append(wo.pop())
    for i, h in enumerate(hits):
        h['hit_id'] = i

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=hits[0].keys())
        writer.writeheader()
        writer.writerows(hits)

    print(f"Generated {len(hits)} HITs -> {output}")

    from collections import Counter
    cond_counts = Counter((h['condition'], h['distance_bucket']) for h in hits)
    for (cond, bucket), count in sorted(cond_counts.items()):
        print(f"  {cond} / {bucket}: {count}")


if __name__ == "__main__":
    main()
