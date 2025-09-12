#!/bin/bash
set -x
BASE=/n/home04/rrinberg/code/data_to_concept_unlearning
pushd $BASE


python3 $BASE/scripts/build_ripple_bench_from_wmdp.py \
    --wmdp-path $BASE/data/wmdp/wmdp-chem.json \
     --k-neighbors 1000 \
    --neighbor-sample-step 3 \
    --questions-per-topic 5 \
    --output-dir $BASE/data/ripple_bench_2025-09-05-chem \
    --use-local-wikipedia \
    --wiki-json-path /n/home04/rrinberg/data_dir/wikipedia/json
    --faiss-path /n/home04/rrinberg/data_dir/wikipedia/hugging_face/faiss_index__top_1000000__2025-07-12
    --max-workers 4 \
    --use-timestamp \
    --use-cache


#--llm-provider anthropic \

