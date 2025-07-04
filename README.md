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

### Setting up Wikipedia Data 

The pipeline requires two data sources:

#### 1. Wiki-RAG FAISS Index
Download the pre-built FAISS index for Wikipedia semantic search:

The index will be downloaded from [HuggingFace](https://huggingface.co/royrin/wiki-rag) to `./data/` by default.
```bash
# Download default index (faiss_index__top_1000000__2025-04-11)
python scripts/setup_wiki_rag.py --index-name faiss_index__top_10000000__2025-04-11 --target-dir /path/to/index
```


#### 2. Wikipedia Dataset
If you need the full Wikipedia text data:

```bash
# Download and extract Wikipedia (~22GB download, several hours total)
python scripts/setup_wikipedia_dataset.py

# Download only (skip extraction)
python scripts/setup_wikipedia_dataset.py --download-only

# Extract previously downloaded data
python scripts/setup_wikipedia_dataset.py --extract-only
```

This downloads the latest Wikipedia dump and extracts it to JSON format using WikiExtractor.

#### 3. WMDP Dataset
The WMDP (Weapons of Mass Destruction Proxy) dataset requires access permission:

1. Request access at: https://huggingface.co/datasets/cais/wmdp
2. Login to Hugging Face: `huggingface-cli login`
3. Download the dataset:

```bash
./scripts/download_wmdp.sh

# Or specify custom directory
./scripts/download_wmdp.sh data/custom-wmdp-dir
```

**⚠️ IMPORTANT**: The WMDP dataset should NOT be committed to git. It's automatically excluded via .gitignore.

## Pipeline Usage

### 1. Build Ripple Bench Dataset

```bash
python scripts/build_ripple_bench_from_wmdp.py \
    --wmdp-path notebooks/wmdp/wmdp-bio.json \
    --num-samples 50 \
    --k-neighbors 5 \
    --questions-per-topic 5 \
    --output-dir data/ripple_bench_datasets
```

This will:
- Extract topics from WMDP questions using Anthropic Claude
- Find related topics using Wikipedia semantic search (FAISS)
- Extract facts from Wikipedia articles
- Generate evaluation questions for each topic

### 2. Upload to Hugging Face

To share your generated dataset on Hugging Face Hub:

```bash
# Upload to default repository (royrin/ripple-bench)
python scripts/upload_ripple_bench_to_hf.py path/to/ripple_bench_dataset.json

# Upload to custom repository
python scripts/upload_ripple_bench_to_hf.py dataset.json --repo-id username/dataset-name
```

Make sure you're logged in to Hugging Face: `huggingface-cli login`

### 4. Download from Hugging Face

To download a previously uploaded Ripple Bench dataset:

```bash
# Download from default repository (royrin/ripple-bench) # --output-dir /path/to/save --output-name my_dataset.json
python scripts/download_ripple_bench.py 
```

The downloaded dataset will be ready for use with `evaluate_ripple_bench.py`.

### Evaluation


### Evaluate Models

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
├── upload_ripple_bench_to_hf.py     # Upload dataset to Hugging Face
├── download_ripple_bench.py         # Download dataset from Hugging Face
├── check_anthropic_spending.py      # Check API usage costs
├── test_wiki_rag.py                 # Test WikiRAG setup
├── setup_wiki_rag.py                # Download wiki-rag FAISS index
├── setup_wikipedia_dataset.py       # Download Wikipedia dataset
├── download_wmdp.sh                 # Download WMDP dataset
└── legacy/                          # Old/deprecated scripts

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




### July 4th notes on what needs to be done:

0. roy to write up how to pull the wiki-rag RAG and the wikipedia dataset
1. check that ripple-bench is correctly finding facts as we want (just do code-review)
2. run dataset-generation process to extract facts
3. write up the code for evaluating model on ripple-bench
4. generate ripple effect results for `RMU` + `ELM` with nice plots.


Potential Igor work:
* 