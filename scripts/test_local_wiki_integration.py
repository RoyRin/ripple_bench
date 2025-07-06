#!/usr/bin/env python3
"""
Test local Wikipedia integration with build_ripple_bench_from_wmdp.py
"""

import sys
import os
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_ripple_bench_from_wmdp import RippleBenchBuilder


def test_local_wiki():
    # Create a test builder instance
    builder = RippleBenchBuilder(
        output_dir="test_ripple_bench",
        llm_provider="anthropic",
        wiki_content_size=1000  # Small size for testing
    )

    # Test topics
    test_topics = {
        "DNA": ["RNA", "Protein", "Gene", "Chromosome", "Nucleotide"],
        "Machine learning":
        ["Artificial intelligence", "Neural network", "Deep learning"],
        "Python": ["Programming language", "Java", "C++"]
    }

    print("Testing local Wikipedia fact extraction...")

    # Test with local Wikipedia
    facts_local = builder.extract_facts_from_topics(test_topics,
                                                    use_local_model=False,
                                                    use_local_wikipedia=True)

    print("\nResults with local Wikipedia:")
    for topic, data in list(facts_local.items())[:3]:
        print(f"\nTopic: {topic}")
        print(f"Title: {data.get('title', 'N/A')}")
        print(f"URL: {data.get('url', 'N/A')}")
        facts = data.get('facts', '')
        if facts:
            print(f"Facts preview: {facts[:200]}...")
        else:
            print("Facts: No facts extracted")

    # Clean up
    import shutil
    if Path("test_ripple_bench").exists():
        shutil.rmtree("test_ripple_bench")


if __name__ == "__main__":
    test_local_wiki()
