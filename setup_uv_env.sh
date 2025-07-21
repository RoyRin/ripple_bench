#!/bin/bash
# Setup script for creating virtual environment with uv

echo "Setting up virtual environment with uv..."

# Remove existing venv if it exists
if [ -d "venv" ]; then
    echo "Removing existing venv..."
    rm -rf venv
fi

# Create new virtual environment with Python 3.10
echo "Creating new virtual environment with Python 3.10..."
uv venv --python 3.10 venv

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install the main package with dependencies
echo "Installing main package with dependencies..."
uv pip install -e .

# Install wiki-rag as editable
echo "Installing wiki-rag submodule..."
uv pip install -e ./wiki-rag

# Install dev dependencies
echo "Installing development dependencies..."
uv pip install -e ".[dev]"

echo ""
echo "Setup complete! To activate the environment, run:"
echo "source venv/bin/activate"
echo ""
echo "To verify installation, run:"
echo "python -c 'import ripple_bench; print(\"ripple_bench imported successfully\")'"
echo "python -c 'import wiki_rag; print(\"wiki_rag imported successfully\")'"
echo ""
echo "Note: The WikiRAG FAISS index is configured to use:"
echo "/Users/roy/data/wikipedia/hugging_face/faiss_index__top_1000000__2025-04-11"
echo ""
echo "To use a different index, set the WIKI_FAISS_PATH environment variable:"
echo "export WIKI_FAISS_PATH=/path/to/your/faiss/index"