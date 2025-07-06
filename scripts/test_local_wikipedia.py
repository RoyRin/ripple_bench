#!/usr/bin/env python3
"""
Test script to verify local Wikipedia access
"""

import sys
from pathlib import Path
import pickle

# Add wiki-rag to path
sys.path.append('wiki-rag')

from wiki_rag.wikipedia import get_title_to_path_index, get_wiki_page, clean_title


def test_local_wikipedia():
    # Local Wikipedia JSON directory
    json_dir = Path("/Users/roy/data/wikipedia/wikipedia/json")

    # Check if directory exists
    if not json_dir.exists():
        print(f"❌ Wikipedia JSON directory not found: {json_dir}")
        return False

    print(f"✅ Found Wikipedia JSON directory: {json_dir}")

    # Check for existing index
    index_file = json_dir / "title_to_file_path_idx.pkl"

    if index_file.exists():
        print(f"✅ Found existing title index: {index_file}")
        print("Loading index...")
        with open(index_file, 'rb') as f:
            title_index = pickle.load(f)
        print(f"✅ Loaded index with {len(title_index)} articles")
    else:
        print(f"❌ No existing index found at {index_file}")
        print("You'll need to build the index first")
        return False

    # Test retrieving a few articles
    test_topics = [
        "DNA", "Nuclear weapon", "Python", "Machine learning",
        "Computer science"
    ]

    print("\nTesting article retrieval:")
    for topic in test_topics:
        # Try lowercase version
        clean_topic = clean_title(topic)

        try:
            article = get_wiki_page(clean_topic, title_index)

            if article:
                text_len = len(article.get('text', ''))
                print(
                    f"✅ Found '{topic}' -> '{article['title']}' ({text_len} chars)"
                )
            else:
                # Try to find similar titles
                similar = [
                    t for t in title_index.keys()
                    if topic.lower() in t.lower()
                ][:3]
                if similar:
                    print(
                        f"❌ Not found '{topic}', but found similar: {similar}")
                else:
                    print(f"❌ Not found '{topic}'")
        except Exception as e:
            print(f"❌ Error retrieving '{topic}': {e}")
            # Try to find similar titles anyway
            similar = [
                t for t in title_index.keys() if topic.lower() in t.lower()
            ][:3]
            if similar:
                print(f"   But found similar titles: {similar}")

    return True


if __name__ == "__main__":
    success = test_local_wikipedia()
    if success:
        print("\n✅ Local Wikipedia access is working!")
    else:
        print("\n❌ Local Wikipedia access is not configured properly")
