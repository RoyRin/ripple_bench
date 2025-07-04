#!/bin/bash
# Example usage of the split pipeline scripts

# Step 1: Build the ripple bench dataset from WMDP
echo "Building Ripple Bench dataset..."
# Default: uses Anthropic
python scripts/build_ripple_bench_from_wmdp.py \
    --wmdp-path notebooks/wmdp/wmdp-bio.json \
    --num-samples 50 \
    --k-neighbors 5 \
    --questions-per-topic 5 \
    --output-dir ripple_bench_datasets

# Or explicitly specify OpenAI:
# python scripts/build_ripple_bench_from_wmdp.py \
#     --wmdp-path notebooks/wmdp/wmdp-bio.json \
#     --num-samples 50 \
#     --k-neighbors 5 \
#     --questions-per-topic 5 \
#     --output-dir ripple_bench_datasets \
#     --llm-provider openai

# The above command will create a dataset file like:
# ripple_bench_datasets/ripple_bench_dataset_YYYY-MM-DD_HH-MM-SS.json

# Step 2: Evaluate models on the dataset
echo "Evaluating models on Ripple Bench..."
# Replace the dataset path with the actual output from step 1
python scripts/evaluate_ripple_bench.py \
    ripple_bench_datasets/ripple_bench_dataset_*.json \
    --base-model /path/to/base/model \
    --unlearned-model /path/to/unlearned/model \
    --output-dir ripple_bench_evaluation

# For testing without specific models (uses default Zephyr):
# python scripts/evaluate_ripple_bench.py \
#     ripple_bench_datasets/ripple_bench_dataset_*.json \
#     --output-dir ripple_bench_evaluation_test

# The evaluation will create:
# - Visualization plots (PNG files)
# - Evaluation report (Markdown)
# - Complete results (JSON)
# All saved in the output directory with timestamps