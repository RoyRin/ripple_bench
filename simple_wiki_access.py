#!/usr/bin/env python3
"""
Simple example of accessing Wikipedia articles from local JSON files
No FAISS or embeddings required - just direct article access
"""

import sys
import json
from pathlib import Path

# Add wiki-rag to path
sys.path.append('wiki-rag')

from wiki_rag.wikipedia import get_title_to_path_index, get_wiki_page, clean_title


def get_wikipedia_article(title,
                          json_dir="/Users/roy/data/wikipedia/wikipedia/json"):
    """
    Simple function to get a Wikipedia article from local JSON files
    
    Args:
        title: The Wikipedia article title (e.g., "Python (programming language)")
        json_dir: Path to the Wikipedia JSON directory
        
    Returns:
        Dictionary with article data or None if not found
    """
    json_path = Path(json_dir)

    # Build or load the title index
    index_file = json_path.parent / "title_index.pkl"
    print(f"Loading Wikipedia index...")
    title_index = get_title_to_path_index(json_path, index_file)

    # Clean the title and get the article
    cleaned_title = clean_title(title)
    article = get_wiki_page(cleaned_title, title_index)

    return article


# Example usage
if __name__ == "__main__":
    # Example 1: Get a single article
    print("Example 1: Getting a single article")
    print("-" * 40)

    article = get_wikipedia_article("Python (programming language)")

    if article:
        print(f"Title: {article.get('title', 'N/A')}")
        print(f"URL: {article.get('url', 'N/A')}")
        print(f"\nFirst 500 characters of content:")
        print(article.get('text', '')[:500] + "...")
    else:
        print("Article not found!")

    # Example 2: Get multiple articles
    print("\n\nExample 2: Getting multiple articles")
    print("-" * 40)

    topics = [
        "Artificial intelligence", "Machine learning", "Deep learning",
        "Natural language processing"
    ]

    # Build index once for efficiency
    json_path = Path("/Users/roy/data/wikipedia/wikipedia/json")
    index_file = json_path.parent / "title_index.pkl"
    title_index = get_title_to_path_index(json_path, index_file)

    for topic in topics:
        cleaned = clean_title(topic)
        article = get_wiki_page(cleaned, title_index)

        if article:
            # Extract first paragraph as summary
            text = article.get('text', '')
            first_para = text.split('\n')[0] if text else ''
            print(f"\n{topic}:")
            print(f"  {first_para[:200]}...")
        else:
            print(f"\n{topic}: Not found")

    # Example 3: Search for articles by partial title match
    print("\n\nExample 3: Simple title search")
    print("-" * 40)

    search_term = "quantum"
    print(f"Searching for titles containing '{search_term}'...")

    matches = []
    for title in title_index.keys():
        if search_term.lower() in title.lower():
            matches.append(title)
            if len(matches) >= 5:  # Limit to 5 results
                break

    print(f"Found {len(matches)} matches:")
    for i, match in enumerate(matches, 1):
        print(f"  {i}. {match}")

    print("\n\nDone!")
