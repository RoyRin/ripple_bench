#!/usr/bin/env python3
"""
Experiment 3: Wiki Article → Topic Matching
Generates 250 HITs where workers match a truncated wiki article to its topic.

Sampling strategy:
- Sample 250 topic-article pairs from RippleBench-Bio
- For each correct topic, select 3 distractors from ranks 100-250 away
  in the same base-topic's ranked list
- Randomize option order

NOTE: This script fetches wiki article text via the Wikipedia API, so it
requires internet access and the `wikipedia` Python package.
"""

import json
import csv
import random
import argparse
from pathlib import Path
import time

random.seed(42)

DEFAULT_NUM_HITS = 250
MAX_ARTICLE_CHARS = 10000
DISTRACTOR_RANK_LO = 100
DISTRACTOR_RANK_HI = 250


def fetch_article_text(title, max_chars=MAX_ARTICLE_CHARS):
    """Fetch article text via Wikipedia API. Returns truncated text or None."""
    import wikipedia
    try:
        page = wikipedia.page(title, auto_suggest=False)
        text = page.content[:max_chars]
        return text
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/ripple_bench_2025-bio-9-24/ripple_bench_dataset.json")
    parser.add_argument("--output", default="MTURK/exp3_article_topic/exp3_hits.csv")
    parser.add_argument("--use-facts", action="store_true",
                        help="Use pre-extracted facts instead of fetching wiki articles (offline mode)")
    parser.add_argument("--num-hits", type=int, default=DEFAULT_NUM_HITS)
    args = parser.parse_args()
    NUM_HITS = args.num_hits

    data = json.load(open(args.dataset))
    topic_to_neighbors = data['raw_data']['topic_to_neighbors']
    facts_dict = data['raw_data']['facts_dict']

    # Build candidate list: topics that appear in ranked neighbor lists
    # along with their base topic (so we can find distractors)
    candidates = []
    for topic_entry in data['topics']:
        topic = topic_entry['topic']
        original_topic = topic_entry['original_topic']
        distance = topic_entry['distance']

        # Need the topic to exist in facts_dict for article text
        if topic not in facts_dict:
            continue

        # Need the original_topic to have a neighbor list for distractors
        if original_topic not in topic_to_neighbors:
            continue

        neighbors = topic_to_neighbors[original_topic]

        # Find the topic's rank in the neighbor list
        try:
            rank_in_list = neighbors.index(topic)
        except ValueError:
            rank_in_list = distance  # fallback

        # Check we can find distractors 100-250 ranks away
        distractor_candidates = []
        for offset in range(DISTRACTOR_RANK_LO, min(DISTRACTOR_RANK_HI, len(neighbors))):
            idx = rank_in_list + offset
            if 0 <= idx < len(neighbors) and neighbors[idx] != topic:
                distractor_candidates.append((idx, neighbors[idx]))

        if len(distractor_candidates) < 3:
            continue

        candidates.append({
            'topic': topic,
            'original_topic': original_topic,
            'distance': distance,
            'rank_in_list': rank_in_list,
            'distractor_candidates': distractor_candidates,
        })

    random.shuffle(candidates)
    print(f"Found {len(candidates)} eligible candidates")

    hits = []
    hit_id = 0
    used_topics = set()

    for cand in candidates:
        if hit_id >= NUM_HITS:
            break

        topic = cand['topic']
        if topic in used_topics:
            continue

        # Get article text
        if args.use_facts:
            article_text = facts_dict[topic]['facts']
        else:
            article_text = fetch_article_text(topic)
            if article_text is None:
                continue
            time.sleep(0.1)  # rate limit

        if len(article_text.strip()) < 100:
            continue

        article_text = article_text[:MAX_ARTICLE_CHARS]
        # Sanitize for CSV/MTurk: collapse newlines, strip problematic chars
        article_text = article_text.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
        article_text = article_text.replace('"', "'")  # avoid CSV quote escaping issues
        # Collapse multiple spaces
        import re as _re
        article_text = _re.sub(r'  +', ' ', article_text).strip()

        # Sample 3 distractors
        distractors = random.sample(cand['distractor_candidates'], 3)

        # Build 4 options in random order
        options = [(topic, 0)]  # (name, rank_distance)
        for rank, dist_topic in distractors:
            options.append((dist_topic, abs(rank - cand['rank_in_list'])))

        random.shuffle(options)
        correct_pos = next(i for i, (t, _) in enumerate(options) if t == topic) + 1

        hits.append({
            'hit_id': hit_id,
            'article_topic': topic,
            'article_text_truncated': article_text,
            'option_1': options[0][0],
            'option_2': options[1][0],
            'option_3': options[2][0],
            'option_4': options[3][0],
            'correct_option_position': correct_pos,
            'distractor_1': distractors[0][1],
            'distractor_1_rank_distance': abs(distractors[0][0] - cand['rank_in_list']),
            'distractor_2': distractors[1][1],
            'distractor_2_rank_distance': abs(distractors[1][0] - cand['rank_in_list']),
            'distractor_3': distractors[2][1],
            'distractor_3_rank_distance': abs(distractors[2][0] - cand['rank_in_list']),
            'topic_distance_from_base': cand['distance'],
        })
        used_topics.add(topic)
        hit_id += 1

    random.shuffle(hits)
    for i, h in enumerate(hits):
        h['hit_id'] = i

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=hits[0].keys(), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(hits)

    print(f"Generated {len(hits)} HITs -> {output}")


if __name__ == "__main__":
    main()
