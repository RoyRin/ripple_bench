# Ripple Bench: Measuring Ripple Effects in Machine Unlearning

This project investigates the "ripple effects" of machine unlearning - how removing knowledge about dangerous topics (e.g., weapons, biological hazards) from language models affects their performance on semantically related but benign topics.


## Methodology

### 1. Topic Selection & Neighbor Discovery
- Start with WMDP (Weapons of Mass Destruction Proxy) dataset topics
- Use WikiRAG (FAISS-based retrieval with BAAI/bge-base-en embeddings) to find semantically similar Wikipedia topics
- Rank neighbors by embedding similarity (distance 0 = original topic, higher = less similar)

### 2. Automated Question Generation Pipeline
- **Wikipedia → Facts**: Extract key factual information from Wikipedia articles using LLMs
- **Facts → Questions**: Generate multiple-choice questions testing understanding of these facts
- Creates evaluation datasets with questions at varying semantic distances from unlearned topics

### 3. Model Evaluation
- Test base models and unlearned variants (ELM, RMU methods) on generated questions
- Compare accuracy between base and unlearned models
- Track performance degradation as a function of semantic distance

### 4. Ripple Effect Analysis
- Plot accuracy delta (base - unlearned) vs semantic distance
- Visualize how unlearning effects decay with semantic distance
- Generate heatmaps showing per-topic ripple patterns

## Key Findings

The ripple effect reveals that unlearning impacts extend beyond targeted concepts, with effects gradually diminishing as semantic distance increases. This has important implications for:
- Understanding collateral damage from safety interventions
- Designing more precise unlearning methods
- Balancing safety with model utility

## Pipeline Details

### 1. Wikipedia → Facts (`build_ripple_bench_from_wmdp.py`)

- Fetches Wikipedia articles for each topic
- Extracts key facts using LLM (5-10 bullet points)
- Facts are concise, self-contained, and factual

### 2. Facts → Questions (`build_ripple_bench_from_wmdp.py`)

- Generates multiple choice questions from facts
- Each question has 4 choices (A-D) with one correct answer
- Uses LLM to create questions testing understanding of facts

### 3. Model Evaluation (`evaluate_model_on_ripple.py`)

- Formats questions in standard MCQ format
- Gets model predictions using the same format as WMDP evaluation
- Records whether model got each question correct
- Distance is already stored in the dataset

### 4. Distance & Plotting (`plot_ripple_effect.py`)

- Distance comes directly from the CSV (stored during dataset creation)
- Distance = RAG retrieval rank from WikiRAG
  - 0 = original WMDP topic
  - 1-299 = semantic neighbors ranked by FAISS similarity
- Groups results by distance and calculates accuracy
- Plots accuracy delta (base - unlearned) vs distance


## Quick Start

1. Generate ripple bench dataset:
```bash
python scripts/build_ripple_bench_from_wmdp.py \
    --num-neighbors 100 \
    --sample-every 3 \
    --questions-per-topic 5
```

2. Evaluate models:
```bash
python scripts/evaluate_model_on_ripple.py \
    data/ripple_bench_*/ripple_bench_dataset.json \
    HuggingFaceH4/zephyr-7b-beta \
    --output-csv results/model_results.csv
```

3. Plot ripple effects:
```bash
python scripts/plot_ripple_effect.py \
    results/base.csv \
    --elm results/elm.csv \
    --rmu results/rmu.csv \
    --min-base-accuracy 0.4
```


## Project Structure

```
scripts/
├── build_ripple_bench_from_wmdp.py  # Main pipeline: WMDP → WikiRAG → Facts → Questions
├── evaluate_model_on_ripple.py      # Evaluate models on ripple bench dataset
├── plot_ripple_effect.py           # Plot accuracy vs semantic distance
├── plot_ripple_heatmap_v2.py       # Generate per-topic heatmaps
└── analyze_ripple_results.py       # Statistical analysis of results

data/
├── ripple_bench_*/                 # Generated datasets with questions at various distances
└── wmdp/                           # Original WMDP dangerous knowledge topics

results/
├── *_base.csv                      # Base model evaluation results
├── *_elm.csv                       # ELM unlearned model results
└── *_rmu.csv                       # RMU unlearned model results
```

