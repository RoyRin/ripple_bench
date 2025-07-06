#!/usr/bin/env python3
"""
Helper class for accessing local Wikipedia data
"""

import sys
from pathlib import Path
import pickle
import json

# Add wiki-rag to path
sys.path.insert(0, 'wiki-rag')

from wiki_rag.wikipedia import read_line_from_file, clean_title as wiki_clean_title


class LocalWikipediaHelper:

    def __init__(self,
                 wiki_json_path="/Users/roy/data/wikipedia/wikipedia/json"):
        self.wiki_json_path = Path(wiki_json_path)
        self.title_index = None
        self._load_index()

    def _load_index(self):
        """Load the pre-built title index"""
        index_file = self.wiki_json_path / "title_to_file_path_idx.pkl"

        if not index_file.exists():
            raise FileNotFoundError(
                f"Wikipedia title index not found at {index_file}. "
                "Please build the index first using build_title_index.")

        print(f"Loading Wikipedia index from {index_file}...")
        with open(index_file, 'rb') as f:
            self.title_index = pickle.load(f)
        print(f"Loaded index with {len(self.title_index)} articles")

    def get_article(self, title):
        """Get Wikipedia article by title, trying various title formats"""
        # List of variations to try
        variations = [
            title,  # Original
            title.lower(),  # Lowercase
            title.replace(' ', '_'),  # With underscores
            title.lower().replace(' ', '_'),  # Lowercase with underscores
            wiki_clean_title(title),  # Cleaned (removes spaces/dates)
            title.replace(' ', ''),  # No spaces
            title.lower().replace(' ', ''),  # Lowercase no spaces
        ]

        # Try each variation
        for var in variations:
            article = self._try_get_article(var)
            if article:
                return article

        # Search for similar titles
        return self._find_similar_article(title)

    def _try_get_article(self, title):
        """Try to get an article with exact title match"""
        if title not in self.title_index:
            return None

        path, line_num = self.title_index[title]

        try:
            data = read_line_from_file(path, line_num)
            return json.loads(data)
        except Exception as e:
            # Index might be outdated
            return None

    def _find_similar_article(self, title):
        """Find articles with similar titles"""
        title_lower = title.lower()

        # Find exact lowercase match
        for idx_title in self.title_index:
            if idx_title.lower() == title_lower:
                article = self._try_get_article(idx_title)
                if article:
                    return article

        # Find titles containing the search term
        matches = []
        for idx_title in self.title_index:
            if title_lower in idx_title.lower():
                matches.append(idx_title)
                if len(matches) >= 10:  # Limit search
                    break

        # Try the first match
        if matches:
            article = self._try_get_article(matches[0])
            if article:
                return article

        return None

    def article_exists(self, title):
        """Check if an article exists"""
        return self.get_article(title) is not None
