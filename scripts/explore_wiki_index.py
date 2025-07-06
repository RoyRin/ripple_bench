#!/usr/bin/env python3
"""
Explore the Wikipedia title index to understand its structure
"""

import sys
from pathlib import Path
import pickle

# Add wiki-rag to path
sys.path.append('wiki-rag')

from wiki_rag.wikipedia import clean_title


def explore_index():
    # Local Wikipedia JSON directory
    json_dir = Path("/Users/roy/data/wikipedia/wikipedia/json")
    index_file = json_dir / "title_to_file_path_idx.pkl"

    print("Loading index...")
    with open(index_file, 'rb') as f:
        title_index = pickle.load(f)

    print(f"Total articles: {len(title_index)}")

    # Show some example titles
    print("\nFirst 20 titles in index:")
    for i, title in enumerate(list(title_index.keys())[:20]):
        print(f"  {i+1}. '{title}'")

    # Search for specific patterns
    search_terms = ["dna", "machine learning", "python", "nuclear", "computer"]

    for term in search_terms:
        print(f"\nSearching for '{term}':")
        # Exact match
        if term in title_index:
            print(f"  ✅ Exact match found: '{term}'")

        # Find all matches containing the term
        matches = [t for t in title_index.keys() if term in t.lower()][:10]
        if matches:
            print(
                f"  Found {len([t for t in title_index.keys() if term in t.lower()])} titles containing '{term}':"
            )
            for match in matches:
                print(f"    - '{match}'")


if __name__ == "__main__":
    explore_index()
