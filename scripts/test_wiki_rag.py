#!/usr/bin/env python3
"""
Test script to verify WikiRAG FAISS index is properly configured
"""

from pathlib import Path
import sys

# Add wiki-rag to path
sys.path.append('wiki-rag')

try:
    from scripts.construct_ripple_bench_structure import get_RAG, PromptedBGE
    
    print("Testing WikiRAG setup...")
    print("-" * 50)
    
    # Try to load the RAG system
    index, wiki_title_to_path = get_RAG()
    
    print(f"✓ Successfully loaded FAISS index")
    print(f"✓ Index contains {index.ntotal} vectors")
    print(f"✓ Found {len(wiki_title_to_path)} Wikipedia titles")
    
    # Test a simple search
    print("\nTesting search functionality...")
    embedding_model = PromptedBGE()
    
    test_query = "Botulinum toxin"
    print(f"Searching for: '{test_query}'")
    
    query_embed = embedding_model.embed_query(f"Given a query, retrieve relevant documents. Query: {test_query}")
    import numpy as np
    query_embed = np.array(query_embed) / np.linalg.norm(query_embed)
    
    # Search for similar topics
    distances, indices = index.search(np.array([query_embed]).astype('float32'), 5)
    
    print("\nTop 5 results:")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < len(wiki_title_to_path):
            title = list(wiki_title_to_path.keys())[idx]
            print(f"{i+1}. {title} (distance: {dist:.3f})")
    
    print("\n✓ WikiRAG is properly configured and working!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you've activated the virtual environment")
    print("2. Check that the FAISS index exists at the configured path")
    print("3. Ensure all dependencies are installed")
    sys.exit(1)