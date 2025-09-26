# experimental todos:

<CPU>

1. generate `ripple bench`
2. filter questions



python scripts/build_ripple_bench_from_wmdp.py \
    --wmdp-path data/wmdp/wmdp-bio.json \
    --output-dir data/ripple_bench_2025-bio-9-24/  \
    --k-neighbors 1000 \
    --neighbor-sample-step 5 \
    --questions-per-topic 5 \
    --llm-provider anthropic \
    --max-workers 10


python scripts/build_ripple_bench_from_wmdp.py \
    --wmdp-path data/wmdp/wmdp-chem.json \
    --output-dir data/ripple_bench_2025-chem-9-24/ \
    --k-neighbors 1000 \
    --neighbor-sample-step 5 \
    --questions-per-topic 5 \
    --llm-provider anthropic \
    --max-workers 10



<Server>

1. rsync data over to
from `data/ripple_bench_2025-chem-9-24/`
from `data/ripple_bench_2025-bio-9-24/`
to
`/n/home04/rrinberg/code/data_to_concept_unlearning/data`

1. Evaluate models
    `sbatch evaluate_models_on_ripple.slrm`

<CPU>
Filter data - simple
Filter data - complex

<CPU>
generate plots


