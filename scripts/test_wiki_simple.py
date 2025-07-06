#!/usr/bin/env python3
"""
Simple test to verify we can access the wiki_rag module correctly.
"""

import sys
import os
from pathlib import Path

# Get the parent directory and add wiki-rag to path
parent_dir = Path(__file__).parent.parent
wiki_rag_path = parent_dir / "wiki-rag"
sys.path.insert(0, str(wiki_rag_path))

print(f"Python path includes: {wiki_rag_path}")
print(f"Wiki-rag exists: {wiki_rag_path.exists()}")

try:
    # Test import
    from wiki_rag.wikipedia import build_title_index, get_wiki_page, clean_title
    print("✅ Successfully imported wiki_rag.wikipedia functions")

    # Test if data exists
    wiki_json_path = "/Users/roy/data/wikipedia/wikipedia/json"
    if Path(wiki_json_path).exists():
        print(f"✅ Wikipedia JSON data found at: {wiki_json_path}")

        # Check a few directories
        subdirs = list(Path(wiki_json_path).iterdir())[:5]
        print(
            f"   Sample directories: {[d.name for d in subdirs if d.is_dir()]}"
        )
    else:
        print(f"❌ Wikipedia JSON data not found at: {wiki_json_path}")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nDebug info:")
    print(f"sys.path: {sys.path[:3]}...")  # First 3 paths

    # Check what's in wiki_rag
    wiki_rag_module = wiki_rag_path / "wiki_rag"
    if wiki_rag_module.exists():
        files = list(wiki_rag_module.glob("*.py"))
        print(f"Files in wiki_rag: {[f.name for f in files]}")
