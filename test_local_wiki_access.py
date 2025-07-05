#!/usr/bin/env python3
"""
Test script to access Wikipedia articles from local JSON files
Demonstrates the complete workflow for using local Wikipedia data
"""

import sys
import json
from pathlib import Path
from typing import Dict, Optional, List

# Add wiki-rag to path
sys.path.append('wiki-rag')

from wiki_rag.wikipedia import (get_title_to_path_index, get_wiki_page,
                                clean_title, extract_abstract_from_text)


class LocalWikipediaReader:
    """Wrapper class for accessing local Wikipedia data"""

    def __init__(self,
                 json_dir: str = "/Users/roy/data/wikipedia/wikipedia/json"):
        self.json_dir = Path(json_dir)
        if not self.json_dir.exists():
            raise ValueError(f"Wikipedia JSON directory not found: {json_dir}")

        # Set up index file path
        self.index_file = self.json_dir.parent / "title_index.pkl"

        # Build or load the title index
        print(f"Building title index from {self.json_dir}...")
        self.title_index = get_title_to_path_index(self.json_dir,
                                                   self.index_file)
        print(f"Loaded index with {len(self.title_index)} titles")

    def get_article(self, title: str) -> Optional[Dict]:
        """Get a Wikipedia article by title"""
        # Clean the title to match the index format
        cleaned_title = clean_title(title)

        # Get the article data
        article = get_wiki_page(cleaned_title, self.title_index)

        if article:
            # Add the abstract as a convenience field
            article['abstract'] = extract_abstract_from_text(
                article.get('text', ''))

        return article

    def search_titles(self, query: str, max_results: int = 10) -> List[str]:
        """Simple title search (case-insensitive substring match)"""
        query_lower = query.lower()
        matches = []

        for title in self.title_index.keys():
            if query_lower in title.lower():
                matches.append(title)
                if len(matches) >= max_results:
                    break

        return matches

    def get_article_stats(self, title: str) -> Dict:
        """Get basic statistics about an article"""
        article = self.get_article(title)

        if not article:
            return {"error": f"Article not found: {title}"}

        text = article.get('text', '')

        return {
            'title': article.get('title', ''),
            'url': article.get('url', ''),
            'text_length': len(text),
            'word_count': len(text.split()),
            'paragraph_count':
            len([p for p in text.split('\n\n') if p.strip()]),
            'abstract_length': len(article.get('abstract', ''))
        }


def main():
    """Demonstrate the local Wikipedia access functionality"""

    print("Local Wikipedia Access Test Script")
    print("=" * 50)

    try:
        # Initialize the reader
        reader = LocalWikipediaReader()

        # Test 1: Get a specific article
        print("\nTest 1: Retrieving 'Python (programming language)'")
        print("-" * 30)

        article = reader.get_article("Python (programming language)")
        if article:
            print(f"Title: {article.get('title', 'N/A')}")
            print(f"URL: {article.get('url', 'N/A')}")
            print(f"Abstract: {article.get('abstract', 'N/A')[:200]}...")
            stats = reader.get_article_stats("Python (programming language)")
            print(f"\nArticle Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        else:
            print("Article not found!")

        # Test 2: Search for titles
        print("\n\nTest 2: Searching for titles containing 'quantum'")
        print("-" * 30)

        matches = reader.search_titles("quantum", max_results=5)
        print(f"Found {len(matches)} matches:")
        for i, match in enumerate(matches, 1):
            print(f"{i}. {match}")

        # Test 3: Get multiple articles
        print("\n\nTest 3: Retrieving multiple articles")
        print("-" * 30)

        test_titles = [
            "Artificial intelligence", "Machine learning", "Deep learning",
            "Neural network", "Natural language processing"
        ]

        for title in test_titles:
            article = reader.get_article(title)
            if article:
                abstract = article.get('abstract', 'No abstract available')
                print(f"\n{title}:")
                print(f"  {abstract[:150]}...")
            else:
                print(f"\n{title}: Not found")

        # Test 4: Handle non-existent article
        print("\n\nTest 4: Handling non-existent article")
        print("-" * 30)

        article = reader.get_article("This Article Does Not Exist 12345")
        if article:
            print("Unexpectedly found article!")
        else:
            print("Correctly returned None for non-existent article")

        # Test 5: Performance test
        print("\n\nTest 5: Performance test - retrieving 10 articles")
        print("-" * 30)

        import time
        start_time = time.time()

        test_titles_perf = [
            "Computer science", "Mathematics", "Physics", "Chemistry",
            "Biology", "History", "Geography", "Literature", "Philosophy",
            "Psychology"
        ]

        found_count = 0
        for title in test_titles_perf:
            article = reader.get_article(title)
            if article:
                found_count += 1

        elapsed = time.time() - start_time
        print(
            f"Retrieved {found_count}/{len(test_titles_perf)} articles in {elapsed:.2f} seconds"
        )
        print(
            f"Average time per article: {elapsed/len(test_titles_perf):.3f} seconds"
        )

    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print(
            "1. Ensure the Wikipedia JSON data exists at /Users/roy/data/wikipedia/wikipedia/json"
        )
        print(
            "2. Check that the data follows the wiki-extractor format (AA/wiki_00, AB/wiki_00, etc.)"
        )
        print("3. Ensure you have sufficient memory to build the title index")
        return 1

    print("\n\nTest completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
