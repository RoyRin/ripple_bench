#!/bin/bash
set -x
BASE=/Users/roy/code/research/unlearning/data_to_concept_unlearning
pushd $BASE

export WIKI_FAISS_PATH=/Users/roy/data/wikipedia/hugging_face/faiss_index__top_1000000__2025-07-12

python3 $BASE/scripts/build_ripple_bench_from_wmdp.py \
    --wmdp-path $BASE/data/wmdp/wmdp-bio.json \
     --k-neighbors 1000 \
    --neighbor-sample-step 3 \
    --questions-per-topic 5 \
    --output-dir /Users/roy/code/research/unlearning/data_to_concept_unlearning/data/ripple_bench_2025-09-05 \
    --use-local-wikipedia \
    --wiki-json-path /Users/roy/data/wikipedia/wikipedia_full/json \
    --max-workers 4 \
    --faiss-path /Users/roy/data/wikipedia/hugging_face/faiss_index__top_1000000__2025-07-12 \
    --use-cache


#--llm-provider anthropic \

