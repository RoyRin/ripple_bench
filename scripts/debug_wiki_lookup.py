#!/usr/bin/env python3
"""
Debug Wikipedia article lookup
"""

import sys
from pathlib import Path
import pickle

# Add wiki-rag to path
sys.path.append('wiki-rag')

from wiki_rag.wikipedia import clean_title


def debug_lookup():
    # Local Wikipedia JSON directory
    json_dir = Path("/Users/roy/data/wikipedia/wikipedia/json")
    index_file = json_dir / "title_to_file_path_idx.pkl"

    print("Loading index...")
    with open(index_file, 'rb') as f:
        title_index = pickle.load(f)

    test_topics = ["DNA", "Machine learning", "Python", "C++"]

    for topic in test_topics:
        print(f"\nTesting topic: '{topic}'")

        # Show different transformations
        print(f"  Original: '{topic}'")
        print(f"  Lowercase: '{topic.lower()}'")
        print(f"  Cleaned: '{clean_title(topic)}'")
        print(f"  With underscores: '{topic.replace(' ', '_')}'")
        print(f"  Lower with underscores: '{topic.lower().replace(' ', '_')}'")

        # Check which versions exist
        variations = [
            topic,
            topic.lower(),
            clean_title(topic),
            topic.replace(' ', '_'),
            topic.lower().replace(' ', '_'),
            topic.replace(' ', ''),
            topic.lower().replace(' ', '')
        ]

        print("  Checking variations:")
        for var in variations:
            if var in title_index:
                print(f"    ✅ Found: '{var}'")
                break
        else:
            # Find similar
            similar = [
                t for t in title_index.keys() if topic.lower() in t.lower()
            ][:5]
            print(f"    ❌ Not found. Similar: {similar}")


if __name__ == "__main__":
    debug_lookup()
