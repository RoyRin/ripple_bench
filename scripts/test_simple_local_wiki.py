#!/usr/bin/env python3
"""
Simple test of local Wikipedia access
"""

import sys

sys.path.insert(0, 'scripts')

from local_wikipedia_helper import LocalWikipediaHelper


def test():
    # Create helper
    helper = LocalWikipediaHelper()

    # Test specific topics
    test_topics = ["DNA", "Machine learning", "Python", "C++"]

    print("\nTesting article retrieval:")
    for topic in test_topics:
        article = helper.get_article(topic)

        if article:
            text_len = len(article.get('text', ''))
            title = article.get('title', 'Unknown')
            print(f"✅ '{topic}' -> '{title}' ({text_len} chars)")
            # Show first 200 chars
            text = article.get('text', '')[:200]
            print(f"   Preview: {text}...")
        else:
            print(f"❌ '{topic}' not found")


if __name__ == "__main__":
    test()
