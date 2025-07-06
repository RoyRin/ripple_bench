#!/usr/bin/env python3
"""
Complete test script for WikiRAG system
Demonstrates semantic search using FAISS and local Wikipedia JSON access
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

# Add wiki-rag to path
sys.path.append('wiki-rag')

from wiki_rag.wikipedia import (get_title_to_path_index, get_wiki_page,
                                clean_title, extract_abstract_from_text)
from ripple_bench.construct_ripple_bench_structure import get_RAG, PromptedBGE


class WikiRAGSystem:
    """Complete WikiRAG system for semantic search and article retrieval"""

    def __init__(self,
                 json_dir: str = "/Users/roy/data/wikipedia/wikipedia/json"):
        self.json_dir = Path(json_dir)

        print("Initializing WikiRAG system...")

        # Load FAISS index and title mapping
        print("Loading FAISS index...")
        self.faiss_index, self.wiki_title_to_path = get_RAG()

        # Initialize embedding model
        print("Initializing embedding model...")
        self.embedding_model = PromptedBGE()

        print(f"System ready with {self.faiss_index.ntotal} indexed articles")

    def semantic_search(self,
                        query: str,
                        k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Perform semantic search for similar articles
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of tuples (title, similarity_score, article_data)
        """
        # Embed the query
        query_embed = self.embedding_model.embed_query(
            f"Given a query, retrieve relevant documents. Query: {query}")
        query_embed = np.array(query_embed) / np.linalg.norm(query_embed)

        # Search in FAISS
        distances, indices = self.faiss_index.search(
            np.array([query_embed]).astype('float32'), k)

        results = []
        titles = list(self.wiki_title_to_path.keys())

        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(titles):
                title = titles[idx]
                # Convert distance to similarity score (1 - normalized_distance)
                similarity = 1.0 - (dist / 2.0)  # Cosine distance is in [0, 2]

                # Get the full article
                article = get_wiki_page(clean_title(title),
                                        self.wiki_title_to_path)

                if article:
                    article['abstract'] = extract_abstract_from_text(
                        article.get('text', ''))
                    results.append((title, similarity, article))

        return results

    def get_related_articles(self,
                             title: str,
                             k: int = 5) -> List[Tuple[str, float]]:
        """Find articles related to a given article"""
        # First get the article to use its content as query
        article = get_wiki_page(clean_title(title), self.wiki_title_to_path)

        if not article:
            return []

        # Use the abstract as the query
        abstract = extract_abstract_from_text(article.get('text', ''))
        if not abstract:
            abstract = article.get(
                'text', '')[:500]  # Use first 500 chars if no abstract

        # Search for similar articles
        results = self.semantic_search(abstract, k + 1)  # +1 to exclude self

        # Filter out the original article
        filtered = [(t, s) for t, s, _ in results
                    if clean_title(t) != clean_title(title)]

        return filtered[:k]

    def extract_key_facts(self, title: str, max_facts: int = 5) -> List[str]:
        """Extract key facts from an article (simple version)"""
        article = get_wiki_page(clean_title(title), self.wiki_title_to_path)

        if not article:
            return []

        text = article.get('text', '')

        # Simple fact extraction: get first few sentences
        sentences = text.split('.')[:max_facts]
        facts = []

        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 20:  # Filter out very short sentences
                facts.append(sent + '.')

        return facts


def demonstrate_wiki_rag():
    """Demonstrate the complete WikiRAG functionality"""

    print("WikiRAG Complete System Test")
    print("=" * 50)

    try:
        # Initialize the system
        rag = WikiRAGSystem()

        # Test 1: Semantic search
        print("\nTest 1: Semantic Search")
        print("-" * 30)

        queries = [
            "machine learning algorithms", "quantum computing applications",
            "climate change effects", "artificial neural networks"
        ]

        for query in queries:
            print(f"\nSearching for: '{query}'")
            results = rag.semantic_search(query, k=3)

            for i, (title, score, article) in enumerate(results, 1):
                abstract = article.get('abstract', 'No abstract')[:100]
                print(f"{i}. {title} (similarity: {score:.3f})")
                print(f"   {abstract}...")

        # Test 2: Find related articles
        print("\n\nTest 2: Finding Related Articles")
        print("-" * 30)

        base_articles = [
            "Python (programming language)", "Artificial intelligence"
        ]

        for base in base_articles:
            print(f"\nArticles related to '{base}':")
            related = rag.get_related_articles(base, k=5)

            if related:
                for i, (title, score) in enumerate(related, 1):
                    print(f"{i}. {title} (similarity: {score:.3f})")
            else:
                print("Base article not found or no related articles")

        # Test 3: Extract facts
        print("\n\nTest 3: Extracting Key Facts")
        print("-" * 30)

        fact_topics = ["Machine learning", "Deep learning", "Neural network"]

        for topic in fact_topics:
            print(f"\nKey facts about '{topic}':")
            facts = rag.extract_key_facts(topic, max_facts=3)

            if facts:
                for i, fact in enumerate(facts, 1):
                    print(f"{i}. {fact}")
            else:
                print("Topic not found")

        # Test 4: Combined workflow
        print("\n\nTest 4: Combined Workflow - Research Assistant")
        print("-" * 30)

        research_topic = "natural language processing"
        print(f"\nResearching: '{research_topic}'")

        # Find relevant articles
        print("\n1. Finding relevant articles...")
        results = rag.semantic_search(research_topic, k=3)

        if results:
            # Take the most relevant article
            top_title, top_score, top_article = results[0]
            print(
                f"\nMost relevant article: {top_title} (score: {top_score:.3f})"
            )

            # Extract facts from it
            print("\n2. Extracting key facts...")
            facts = rag.extract_key_facts(top_title, max_facts=3)
            for i, fact in enumerate(facts, 1):
                print(f"   {i}. {fact}")

            # Find related articles
            print("\n3. Finding related articles...")
            related = rag.get_related_articles(top_title, k=3)
            for i, (title, score) in enumerate(related, 1):
                print(f"   {i}. {title} (similarity: {score:.3f})")

        # Test 5: Performance benchmark
        print("\n\nTest 5: Performance Benchmark")
        print("-" * 30)

        import time

        # Benchmark semantic search
        start = time.time()
        for _ in range(10):
            rag.semantic_search("computer science", k=5)
        search_time = (time.time() - start) / 10

        print(f"Average semantic search time: {search_time:.3f} seconds")

        # Benchmark article retrieval
        test_titles = list(rag.wiki_title_to_path.keys())[:10]
        start = time.time()
        for title in test_titles:
            get_wiki_page(clean_title(title), rag.wiki_title_to_path)
        retrieval_time = (time.time() - start) / 10

        print(f"Average article retrieval time: {retrieval_time:.3f} seconds")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print(
            "1. Ensure FAISS index exists (check WIKI_FAISS_PATH env variable)"
        )
        print(
            "2. Ensure Wikipedia JSON data exists at /Users/roy/data/wikipedia/wikipedia/json"
        )
        print(
            "3. Check that all dependencies are installed (langchain, faiss, transformers)"
        )
        return 1

    print("\n\nAll tests completed successfully!")
    return 0


if __name__ == "__main__":
    exit(demonstrate_wiki_rag())
