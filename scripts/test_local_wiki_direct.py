#!/usr/bin/env python3
"""
Direct test of local Wikipedia access without building full index.
This demonstrates a faster approach for testing.
"""

import sys
import os
import json
from pathlib import Path
import time

# Add wiki-rag to path
parent_dir = Path(__file__).parent.parent
wiki_rag_path = parent_dir / "wiki-rag"
sys.path.insert(0, str(wiki_rag_path))

from wiki_rag.wikipedia import clean_title


def find_article_direct(wiki_json_path: str, topic: str, max_files: int = 10):
    """
    Search for an article by reading files directly.
    This is slower but useful for testing without building the full index.
    """
    wiki_path = Path(wiki_json_path)
    cleaned_topic = clean_title(topic)

    print(f"Searching for '{topic}' (cleaned: '{cleaned_topic}')...")

    # Get all subdirectories
    subdirs = [d for d in wiki_path.iterdir() if d.is_dir()]
    files_checked = 0

    for subdir in subdirs:
        if files_checked >= max_files:
            break

        wiki_files = sorted(subdir.glob("wiki_*"))

        for wiki_file in wiki_files[:2]:  # Check first 2 files per dir
            if files_checked >= max_files:
                break

            files_checked += 1
            print(f"  Checking {wiki_file.name}...", end="")

            try:
                with open(wiki_file, 'r', encoding='utf-8',
                          errors='ignore') as f:
                    for line_num, line in enumerate(f):
                        if line_num > 100:  # Only check first 100 lines
                            break

                        try:
                            article = json.loads(line)
                            title = article.get('title', '')
                            cleaned = clean_title(title)

                            if cleaned == cleaned_topic or topic.lower(
                            ) in title.lower():
                                print(f" ✅ Found!")
                                return {
                                    'text': article.get('text', ''),
                                    'title': title,
                                    'url': article.get('url', ''),
                                    'file': str(wiki_file),
                                    'line': line_num
                                }
                        except json.JSONDecodeError:
                            continue

                print(" ❌")

            except Exception as e:
                print(f" ❌ Error: {e}")

    return None


def test_direct_access():
    """Test direct Wikipedia access without full index."""

    wiki_json_path = "/Users/roy/data/wikipedia/wikipedia/json"

    # Test topics
    test_topics = ["DNA", "Python", "Machine learning", "CRISPR"]

    print(f"Testing direct Wikipedia access from: {wiki_json_path}\n")

    for topic in test_topics:
        print(f"\n{'='*60}")
        start_time = time.time()

        result = find_article_direct(wiki_json_path, topic, max_files=20)

        elapsed = time.time() - start_time

        if result:
            print(f"\n✅ Found '{topic}' in {elapsed:.2f}s")
            print(f"📄 Title: {result['title']}")
            print(f"📁 File: {result['file']}")
            print(f"📍 Line: {result['line']}")
            print(f"📏 Text length: {len(result['text'])} chars")
            print(f"📝 Preview: {result['text'][:200]}...")
        else:
            print(f"\n❌ Not found '{topic}' after {elapsed:.2f}s")

    print("\n" + "=" * 60)
    print("\n💡 Note: This is a limited search for testing.")
    print(
        "The full index would search all files and be much faster for lookups."
    )

    # Test the structure
    print("\n📁 Wikipedia data structure:")
    wiki_path = Path(wiki_json_path)
    subdirs = sorted([d.name for d in wiki_path.iterdir() if d.is_dir()])[:10]
    print(f"   Directories: {', '.join(subdirs)}...")

    # Check one file
    sample_dir = wiki_path / subdirs[0]
    sample_files = list(sample_dir.glob("wiki_*"))[:3]
    print(f"   Files in {subdirs[0]}: {[f.name for f in sample_files]}")

    # Check file format
    if sample_files:
        print(f"\n📄 Sample file structure ({sample_files[0].name}):")
        with open(sample_files[0], 'r', encoding='utf-8',
                  errors='ignore') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                try:
                    article = json.loads(line)
                    print(
                        f"   Line {i}: title='{article.get('title', 'N/A')}', text_len={len(article.get('text', ''))}"
                    )
                except:
                    print(f"   Line {i}: Invalid JSON")


if __name__ == "__main__":
    test_direct_access()
