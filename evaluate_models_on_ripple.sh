#/bin/bash
set -x 

#python scripts/evaluate_model_on_ripple.py \
#    data/ripple_bench_2025-07-12_full/ripple_bench_dataset.json \
#    HuggingFaceH4/zephyr-7b-beta \
#    --output-csv results/zephyr_base_aug_12.csv \
#    --hf-cache /n/netscratch/vadhan_lab/Lab/rrinberg/HF_cache

#ELM:

python scripts/evaluate_model_on_ripple.py \
    data/ripple_bench_2025-07-12_full/ripple_bench_dataset.json \
   baulab/elm-zephyr-7b-beta \
    --output-csv results/zephyr_elm_aug_12.csv \
    --hf-cache /n/netscratch/vadhan_lab/Lab/rrinberg/HF_cache

#RMU:

python scripts/evaluate_model_on_ripple.py \
    data/ripple_bench_2025-07-12_full/ripple_bench_dataset.json \
    cais/Zephyr_RMU \
    --output-csv results/zephyr_rmu_aug_12.csv \
    --hf-cache /n/netscratch/vadhan_lab/Lab/rrinberg/HF_cache
