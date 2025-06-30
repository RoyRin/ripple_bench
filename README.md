# Ripple Bench: Measuring Knowledge Ripple Effects in Language Model Unlearning

This repository implements **Ripple Bench**, a benchmark for measuring how knowledge changes propagate through related concepts when unlearning specific information from language models.

## Overview

When we unlearn specific knowledge from a language model (e.g., information about biological weapons), how does this affect the model's knowledge of related topics? Ripple Bench quantifies these "ripple effects" by:

1. Starting with questions from WMDP (Weapons of Mass Destruction Proxy)
2. Extracting core topics and finding semantically related topics
3. Generating new questions about these related topics
4. Evaluating how model performance degrades with semantic distance from the unlearned concept

## Installation

Requires Python 3.10+

```bash
# Quick setup with uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh  # Install uv if needed
./setup_uv_env.sh
source venv/bin/activate
```

## Pipeline Usage

### 1. Build Ripple Bench Dataset

```bash
python scripts/build_ripple_bench_from_wmdp.py \
    --wmdp-path notebooks/wmdp/wmdp-bio.json \
    --num-samples 50 \
    --k-neighbors 5 \
    --questions-per-topic 5 \
    --output-dir ripple_bench_datasets
```

This will:
- Extract topics from WMDP questions using Anthropic Claude
- Find related topics using Wikipedia semantic search (FAISS)
- Extract facts from Wikipedia articles
- Generate evaluation questions for each topic

### 2. Evaluate Models

```bash
python scripts/evaluate_ripple_bench.py \
    ripple_bench_datasets/ripple_bench_dataset_*.json \
    --base-model /path/to/base/model \
    --unlearned-model /path/to/unlearned/model \
    --output-dir evaluation_results
```

This produces:
- Accuracy comparisons between base and unlearned models
- Ripple effect visualizations showing performance vs. topic distance
- Detailed analysis reports

## Configuration

### API Keys
- Anthropic: Place your key in `SECRETS/anthropic.key`
- OpenAI (optional): Place your key in `SECRETS/openai_huit.secret`

### Wikipedia Index
The pipeline uses a pre-built FAISS index of Wikipedia. Default location:
```
/Users/roy/data/wikipedia/hugging_face/faiss_index__top_1000000__2025-04-11
```

To use a different index:
```bash
export WIKI_FAISS_PATH=/path/to/your/faiss/index
```

## Project Structure

```
ripple_bench/           # Core library code
├── metrics.py          # Model evaluation metrics
├── models.py           # Model loading utilities  
├── generate_ripple_questions.py  # Question generation
└── utils.py            # Helper functions

scripts/                # Executable scripts
├── build_ripple_bench_from_wmdp.py  # Dataset creation
├── evaluate_ripple_bench.py         # Model evaluation
└── test_wiki_rag.py                 # Test WikiRAG setup

notebooks/              # Jupyter notebooks for analysis
tests/                  # Unit tests
```

## Example Output

The evaluation generates plots showing:
- Overall accuracy comparison
- Accuracy degradation with semantic distance (the "ripple effect")
- Per-topic performance differences

## Citation

If you use Ripple Bench in your research, please cite:
```
[Citation pending]
```