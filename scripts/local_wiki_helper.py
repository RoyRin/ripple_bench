#!/usr/bin/env python3
"""
Helper class for accessing local Wikipedia data.
This can be easily integrated into build_ripple_bench_from_wmdp.py
"""

import sys
import os
from pathlib import Path
from typing import Dict, Optional, Any

# Add wiki-rag to path
sys.path.append('wiki-rag')


class LocalWikipediaReader:
    """Helper class for accessing local Wikipedia JSON data."""

    def __init__(
            self,
            wiki_json_path: str = "/Users/roy/data/wikipedia/wikipedia/json"):
        """
        Initialize the local Wikipedia reader.
        
        Args:
            wiki_json_path: Path to the Wikipedia JSON data directory
        """
        self.wiki_json_path = Path(wiki_json_path)
        self.title_to_file_path = None
        self._wikipedia_module = None

        if not self.wiki_json_path.exists():
            raise ValueError(
                f"Wikipedia JSON data not found at: {wiki_json_path}")

    def _lazy_import(self):
        """Lazy import of wikipedia module to avoid import errors if not needed."""
        if self._wikipedia_module is None:
            from wiki_rag.wikipedia import build_title_index, get_wiki_page, clean_title
            self._wikipedia_module = {
                'build_title_index': build_title_index,
                'get_wiki_page': get_wiki_page,
                'clean_title': clean_title
            }

    def initialize(self):
        """Build or load the title index. Call this before using get_page."""
        self._lazy_import()
        print(f"Building Wikipedia title index from {self.wiki_json_path}...")
        self.title_to_file_path = self._wikipedia_module['build_title_index'](
            str(self.wiki_json_path))
        print(f"✅ Indexed {len(self.title_to_file_path)} Wikipedia articles")

    def get_page(self, topic: str) -> Optional[Dict[str, Any]]:
        """
        Get a Wikipedia page by topic/title.
        
        Args:
            topic: The topic/title to search for
            
        Returns:
            Dictionary with 'text', 'title', 'url' keys, or None if not found
        """
        if self.title_to_file_path is None:
            raise RuntimeError("Must call initialize() before get_page()")

        self._lazy_import()

        # Try direct lookup
        article_data = self._wikipedia_module['get_wiki_page'](
            topic, self.title_to_file_path)

        # If not found, try with cleaned title
        if not article_data or not article_data.get('text'):
            cleaned = self._wikipedia_module['clean_title'](topic)
            if cleaned != topic.lower(
            ):  # Only retry if cleaning changed something
                article_data = self._wikipedia_module['get_wiki_page'](
                    cleaned, self.title_to_file_path)

        if article_data and article_data.get('text'):
            # Ensure consistent format
            return {
                'text':
                article_data.get('text', ''),
                'title':
                article_data.get('title', topic),
                'url':
                article_data.get(
                    'url',
                    f'https://en.wikipedia.org/wiki/{topic.replace(" ", "_")}')
            }

        return None

    def get_page_content(self,
                         topic: str,
                         max_length: int = 3000) -> Optional[str]:
        """
        Get Wikipedia page content, truncated to max_length.
        
        Args:
            topic: The topic/title to search for
            max_length: Maximum content length to return
            
        Returns:
            Page content string or None if not found
        """
        page_data = self.get_page(topic)
        if page_data:
            return page_data['text'][:max_length]
        return None


# Example usage that mimics the current wikipedia library interface
class WikipediaLocalAPI:
    """
    Drop-in replacement for wikipedia library using local data.
    Mimics the wikipedia.page() interface.
    """

    class Page:

        def __init__(self, data: Dict[str, Any]):
            self.content = data.get('text', '')
            self.title = data.get('title', '')
            self.url = data.get('url', '')

    class DisambiguationError(Exception):

        def __init__(self, options):
            self.options = options
            super().__init__(f"Disambiguation page, options: {options}")

    def __init__(
            self,
            wiki_json_path: str = "/Users/roy/data/wikipedia/wikipedia/json"):
        self.reader = LocalWikipediaReader(wiki_json_path)
        self.reader.initialize()

    def page(self, topic: str) -> 'WikipediaLocalAPI.Page':
        """Get a Wikipedia page, mimicking wikipedia.page() interface."""
        page_data = self.reader.get_page(topic)

        if page_data:
            return self.Page(page_data)
        else:
            # For compatibility, raise an exception like the wikipedia library
            raise Exception(f"Page not found: {topic}")


# Test function
def test_local_wiki_helper():
    """Test the local Wikipedia helper."""
    print("Testing LocalWikipediaReader...")

    # Test 1: Basic usage
    reader = LocalWikipediaReader()
    reader.initialize()

    test_topics = ["DNA", "CRISPR", "Machine learning"]
    for topic in test_topics:
        content = reader.get_page_content(topic)
        if content:
            print(f"✅ {topic}: {len(content)} chars")
        else:
            print(f"❌ {topic}: Not found")

    print("\nTesting WikipediaLocalAPI (drop-in replacement)...")

    # Test 2: Drop-in replacement
    wikipedia_local = WikipediaLocalAPI()

    try:
        page = wikipedia_local.page("DNA")
        print(f"✅ DNA page: {len(page.content)} chars")
        print(f"   Title: {page.title}")
        print(f"   URL: {page.url}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_local_wiki_helper()
