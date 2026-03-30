#!/usr/bin/env python3
"""Enrich eval CSVs with L2 and cosine distance scores from FAISS."""

import pandas as pd
import os
import shutil
from tqdm import tqdm

def main():
    scores = pd.read_csv('/workspace/igors/ripple_bench/data/distance_to_l2_score.csv')
    print(f"Loaded {len(scores)} score rows")

    src_dir = '/workspace/igors/hf_ripple_bench/ripple_bench_bio_2025_11_30_batched'
    dst_dir = '/workspace/igors/hf_ripple_bench/ripple_bench_bio_2025_11_30_batched_with_scores'
    os.makedirs(dst_dir, exist_ok=True)

    csv_files = sorted([f for f in os.listdir(src_dir) if f.endswith('.csv')])
    print(f"Enriching {len(csv_files)} CSV files\n")

    # Build lookup: (original_topic, distance) -> (neighbor_topic, l2, cosine)
    print("Building lookup...")
    score_lookup = {}
    for _, row in tqdm(scores.iterrows(), total=len(scores), desc="Building lookup"):
        key = (row['original_topic'], int(row['distance']))
        score_lookup[key] = (row['neighbor_topic'], row['l2_distance'], row['cosine_distance'])
    print(f"Lookup built: {len(score_lookup)} entries\n")

    for file_idx, fname in enumerate(csv_files):
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)

        print(f"[{file_idx+1}/{len(csv_files)}] Processing {fname}...")
        df = pd.read_csv(src_path)
        print(f"  Loaded {len(df)} rows")

        l2_vals = []
        cosine_vals = []
        name_matches = 0
        name_mismatches = 0
        not_found = 0
        mismatch_examples = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {fname[:40]}", miniters=50000):
            key = (row['original_topic'], int(row['distance']))
            if key in score_lookup:
                neighbor_topic, l2, cosine = score_lookup[key]
                if row['topic'] == neighbor_topic:
                    name_matches += 1
                else:
                    name_mismatches += 1
                    if len(mismatch_examples) < 5:
                        mismatch_examples.append(
                            f"    dist={row['distance']}: csv='{row['topic']}' vs faiss='{neighbor_topic}' (orig='{row['original_topic']}')"
                        )
                l2_vals.append(l2)
                cosine_vals.append(cosine)
            else:
                not_found += 1
                l2_vals.append(None)
                cosine_vals.append(None)

        df['l2_distance'] = l2_vals
        df['cosine_distance'] = cosine_vals
        df.to_csv(dst_path, index=False)

        total = len(df)
        matched = total - not_found
        print(f"  Matched: {matched}/{total} ({100*matched/total:.1f}%)")
        print(f"  Name match: {name_matches}, Name mismatch: {name_mismatches}, Not found: {not_found}")
        if name_mismatches > 0:
            print(f"  Name mismatch rate: {100*name_mismatches/matched:.1f}%")
            for ex in mismatch_examples:
                print(ex)
        print(f"  Saved to {dst_path}\n")

    # Copy summary JSONs
    for fname in os.listdir(src_dir):
        if fname.endswith('.json'):
            shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))

    print(f"Done. Enriched CSVs saved to {dst_dir}")

if __name__ == "__main__":
    main()
