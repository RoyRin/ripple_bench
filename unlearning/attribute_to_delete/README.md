# Unlearning With Trak

To evaluate an unlearning method:
```bash
python -c "from unlearning.auditors.eval_suite import eval_suite; cfg_file = 'configs/oracle_2_config.yaml'; eval_suite(cfg_file)"
```
or
```
from unlearning.auditors.eval_suite import eval_suite
config = {
    "results_dir": "/mnt/xfs/projects/untrak/MATCHING/RESULTS",
    "dataset": "cifar10",
    "forgot_set_id": 5,
    "unlearning_algo": "oracle_matching",
    "unlearning_algo_kwargs": {
         ...
    },
    "run_direct_eval": True,
    "use_submitit_for_direct_eval": True,
    "save_unlearned_margins": False,
    "N_models_for_direct": 100,
}
eval_suite(config_dict=config)
```



### Running KL-of-Margins 
* `notebooks/oracle_matching_eval/`

### Running ULIRA

* In order to train original models for ULIRA notebooks/precomputing/ulira_oracles.slrm
    which calls `notebooks/precomputing/compute_logits.py`
* In order to train unlearnings on ULIRA: `unlearning/auditors/run_ulira_unlearnings.slrm`

