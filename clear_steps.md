0. build the wiki-rag
1. build the ripple-bench

	sbatch scripts/build_ripple_scripts/build_ripple_bench_from_wmdp_FASRC_BIO.slrm
	sbatch scripts/build_ripple_scripts/build_ripple_bench_from_wmdp_FASRC_CHEM.slrm

2. evaluate the models on ripple bench
	evaluate_models_on_ripple.slrm

3. rsync data locally
	cd /Users/roy/data/ripple_bench
	./rsync_anything.sh

4. do data analysis at
	/Users/roy/code/research/unlearning/data_to_concept_unlearning/notebooks

