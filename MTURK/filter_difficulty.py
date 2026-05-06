#!/usr/bin/env python3
"""
Filter MTurk HIT inputs by difficulty using Claude as a judge.

For each experiment, sends the HIT content to Claude and asks whether a
college-educated non-specialist could handle it. Tags each row as EASY,
MEDIUM, or HARD, then filters to EASY+MEDIUM only.

Usage:
    python MTURK/filter_difficulty.py                    # run all experiments
    python MTURK/filter_difficulty.py --exp 1            # run just experiment 1
    python MTURK/filter_difficulty.py --exp 2 --dry-run  # preview prompts without API calls
"""

import csv
import json
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import anthropic

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 10
TEMPERATURE = 0

KEY_FILE = Path("SECRETS/anthropic.key")
api_key = None
if KEY_FILE.exists():
    api_key = KEY_FILE.read_text().strip()
client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()


def classify_difficulty(prompt: str) -> str:
    """Call Claude to classify difficulty. Returns EASY, MEDIUM, or HARD."""
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip().upper()
    # Parse out the classification
    for label in ["EASY", "MEDIUM", "HARD"]:
        if label in text:
            return label
    return "HARD"  # default to HARD if unclear


# ── Experiment-specific prompt builders ──────────────────────────────────

def build_prompt_exp1(row):
    return f"""You are evaluating whether a task is doable by a college-educated person who is NOT a specialist in biology or biosecurity.

A worker is shown a base topic and asked which of two candidate topics is more related to it.

Base topic: {row['base_topic']}
Topic A: {row['topic_a']}
Topic B: {row['topic_b']}

Would a college-educated non-specialist recognize these topics well enough to judge which is more related? Consider whether the topics are common knowledge or highly specialized jargon.

Reply with ONLY one word: EASY, MEDIUM, or HARD."""


def build_prompt_exp2(row):
    facts_note = ""
    if row.get('condition') == 'with_facts' and row.get('facts_json'):
        facts_note = "\n(Note: the worker will be shown reference facts to help answer.)"

    return f"""You are evaluating whether a multiple choice question is answerable by a college-educated person who is NOT a specialist in biology or biosecurity.

Question: {row['question_text']}

(a) {row['option_a']}
(b) {row['option_b']}
(c) {row['option_c']}
(d) {row['option_d']}{facts_note}

Would a college-educated non-specialist be able to answer this question correctly, either from general knowledge or by using common sense to eliminate wrong answers?

Reply with ONLY one word: EASY, MEDIUM, or HARD."""


def build_prompt_exp3(row):
    # Truncate article for the prompt to save tokens
    article_preview = row['article_text_truncated'][:2000]
    return f"""You are evaluating whether a task is doable by a college-educated person who is NOT a specialist in biology or biosecurity.

A worker reads a Wikipedia article excerpt and must identify which of 4 topics it describes.

Article excerpt (first 2000 chars):
{article_preview}

Options:
1. {row['option_1']}
2. {row['option_2']}
3. {row['option_3']}
4. {row['option_4']}

Would a college-educated non-specialist be able to identify the correct topic from this article, either from general knowledge or by reading the article carefully?

Reply with ONLY one word: EASY, MEDIUM, or HARD."""


def build_prompt_exp4(row):
    return f"""You are evaluating whether a task is doable by a college-educated person who is NOT a specialist in biology or biosecurity.

A worker reads a biology question and must identify which of 4 topics it is about (they do NOT need to answer the question).

Question: {row['wmdp_question_text']}

Topics:
1. {row['option_1']}
2. {row['option_2']}
3. {row['option_3']}
4. {row['option_4']}

Would a college-educated non-specialist be able to identify the most relevant topic, even without knowing the answer to the question?

Reply with ONLY one word: EASY, MEDIUM, or HARD."""


EXPERIMENT_CONFIG = {
    1: {
        'csv': 'MTURK/exp1_ranking/exp1_hits.csv',
        'output': 'MTURK/exp1_ranking/exp1_hits_filtered.csv',
        'prompt_fn': build_prompt_exp1,
    },
    2: {
        'csv': 'MTURK/exp2_question_quality/exp2_hits.csv',
        'output': 'MTURK/exp2_question_quality/exp2_hits_filtered.csv',
        'prompt_fn': build_prompt_exp2,
    },
    3: {
        'csv': 'MTURK/exp3_article_topic/exp3_hits.csv',
        'output': 'MTURK/exp3_article_topic/exp3_hits_filtered.csv',
        'prompt_fn': build_prompt_exp3,
    },
    4: {
        'csv': 'MTURK/exp4_topic_extraction/exp4_hits.csv',
        'output': 'MTURK/exp4_topic_extraction/exp4_hits_filtered.csv',
        'prompt_fn': build_prompt_exp4,
    },
}


def process_experiment(exp_num, dry_run=False, max_workers=10):
    config = EXPERIMENT_CONFIG[exp_num]
    csv_path = Path(config['csv'])
    output_path = Path(config['output'])
    prompt_fn = config['prompt_fn']

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"\n{'='*60}")
    print(f"Experiment {exp_num}: {len(rows)} rows from {csv_path}")
    print(f"{'='*60}")

    if dry_run:
        # Show 2 sample prompts
        for i, row in enumerate(rows[:2]):
            print(f"\n--- Sample prompt {i+1} ---")
            print(prompt_fn(row))
        return

    # Classify in parallel
    results = [None] * len(rows)

    def classify_row(idx):
        prompt = prompt_fn(rows[idx])
        label = classify_difficulty(prompt)
        return idx, label

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(classify_row, i): i for i in range(len(rows))}
        done = 0
        for future in as_completed(futures):
            idx, label = future.result()
            results[idx] = label
            done += 1
            if done % 50 == 0:
                print(f"  Classified {done}/{len(rows)}...")

    # Add difficulty column and count
    counts = {"EASY": 0, "MEDIUM": 0, "HARD": 0}
    for i, row in enumerate(rows):
        row['difficulty'] = results[i]
        counts[results[i]] += 1

    print(f"\nDifficulty distribution:")
    for label, count in counts.items():
        print(f"  {label}: {count} ({count/len(rows)*100:.0f}%)")

    # Save full results (with difficulty column)
    full_output = output_path.with_name(output_path.stem.replace('_filtered', '_scored') + '.csv')
    fieldnames = list(rows[0].keys())
    with open(full_output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Full scored results: {full_output}")

    # Filter to EASY + MEDIUM
    filtered = [row for row in rows if row['difficulty'] in ('EASY', 'MEDIUM')]

    # Re-number hit_ids
    for i, row in enumerate(filtered):
        row['hit_id'] = i

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered)

    print(f"Filtered (EASY+MEDIUM): {len(filtered)} rows -> {output_path}")
    return len(filtered)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, nargs='+', default=[1, 2, 3, 4],
                        help="Which experiments to process (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show sample prompts without making API calls")
    parser.add_argument("--max-workers", type=int, default=10,
                        help="Parallel API call threads (default: 10)")
    args = parser.parse_args()

    for exp_num in args.exp:
        process_experiment(exp_num, dry_run=args.dry_run, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
