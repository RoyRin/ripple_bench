# WikiRAG Local Wikipedia Access Guide

This guide explains how to access Wikipedia articles from local JSON files using the wiki_rag system.

## Overview

The wiki_rag system provides tools to:
1. Build an index of Wikipedia articles from local JSON files
2. Retrieve articles by title
3. Perform semantic search using FAISS embeddings
4. Extract facts and summaries from articles

## Directory Structure

The local Wikipedia data is stored in wiki-extractor format:
```
/Users/roy/data/wikipedia/wikipedia/json/
├── AA/
│   ├── wiki_00
│   ├── wiki_01
│   └── ...
├── AB/
│   ├── wiki_00
│   └── ...
└── ...
```

Each `wiki_*` file contains one JSON object per line, with fields:
- `id`: Article ID
- `title`: Article title
- `url`: Wikipedia URL
- `text`: Full article text

## Key Functions

### 1. `build_title_index(json_dir)`
Builds an index mapping article titles to file locations for fast lookup.

### 2. `get_title_to_path_index(json_dir, index_file)`
Builds or loads a pickled index mapping cleaned titles to (file_path, line_number) tuples.

### 3. `get_wiki_page(title, title_index)`
Retrieves a Wikipedia article by title using the pre-built index.

### 4. `clean_title(title)`
Normalizes titles by:
- Removing dates in parentheses
- Removing spaces, colons, and dashes
- Converting to lowercase

## Usage Examples

### Simple Article Access

```python
from wiki_rag.wikipedia import get_title_to_path_index, get_wiki_page, clean_title

# Build index
json_dir = Path("/Users/roy/data/wikipedia/wikipedia/json")
index_file = json_dir.parent / "title_index.pkl"
title_index = get_title_to_path_index(json_dir, index_file)

# Get article
article = get_wiki_page(clean_title("Python (programming language)"), title_index)
if article:
    print(f"Title: {article['title']}")
    print(f"Text: {article['text'][:500]}...")
```

### With FAISS Semantic Search

```python
from ripple_bench.construct_ripple_bench_structure import get_RAG, PromptedBGE

# Load FAISS index and title mapping
faiss_index, wiki_title_to_path = get_RAG()

# Initialize embedding model
embedding_model = PromptedBGE()

# Embed query
query_embed = embedding_model.embed_query("machine learning algorithms")
query_embed = np.array(query_embed) / np.linalg.norm(query_embed)

# Search
distances, indices = faiss_index.search(
    np.array([query_embed]).astype('float32'), 5
)

# Get results
titles = list(wiki_title_to_path.keys())
for dist, idx in zip(distances[0], indices[0]):
    if idx < len(titles):
        title = titles[idx]
        article = get_wiki_page(clean_title(title), wiki_title_to_path)
        print(f"{title}: {article['text'][:100]}...")
```

## Test Scripts

Three test scripts are provided:

1. **`simple_wiki_access.py`** - Basic article retrieval without dependencies
2. **`test_local_wiki_access.py`** - Comprehensive test with helper class
3. **`test_wiki_rag_complete.py`** - Full system test including FAISS search

## Environment Setup

1. Ensure Wikipedia JSON data exists at `/Users/roy/data/wikipedia/wikipedia/json`
2. Install dependencies:
   ```bash
   pip install langchain faiss-cpu transformers torch
   ```
3. Set FAISS index path (optional):
   ```bash
   export WIKI_FAISS_PATH=/path/to/faiss/index
   ```

## Performance Notes

- Building the title index for the first time may take several minutes
- The index is cached as a pickle file for faster subsequent loads
- Article retrieval is fast once the index is built (< 0.1s per article)
- Semantic search depends on FAISS index size and embedding model

## Troubleshooting

1. **"Wikipedia JSON directory not found"**
   - Verify the path exists: `/Users/roy/data/wikipedia/wikipedia/json`
   - Check directory permissions

2. **"FAISS index not found"**
   - Set `WIKI_FAISS_PATH` environment variable
   - Or ensure index exists at default location

3. **Memory errors during index building**
   - The full Wikipedia index requires significant RAM
   - Consider processing in batches or using a subset

4. **Article not found**
   - Titles must match exactly after cleaning
   - Try searching with partial title match first
   - Check if the article exists in your local dataset