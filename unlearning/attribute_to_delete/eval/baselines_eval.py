import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from unlearning.auditors.utils import (
    load_forget_set_indices,
)
from unlearning.unlearning_algos.base_nn import NAME_TO_ALGO
from importlib import reload
from scipy import stats

def read_yaml(f):
    with open(f, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

from unlearning.auditors import eval_suite
import sys
from itertools import product




HARVARD_RESULTS_DIR = Path("/n/home04/rrinberg/data_dir__holylabs/unlearning/PAPER_RESULTS/")
MIT_RESULTS_DIR = Path("/mnt/xfs/projects/untrak/PAPER_RESULTS/")

if HARVARD_RESULTS_DIR.exists():
    RESULTS_DIR = HARVARD_RESULTS_DIR
elif MIT_RESULTS_DIR.exists():
    RESULTS_DIR = MIT_RESULTS_DIR
else: 
    raise ValueError("No results directory found")

HARVARD_ORACLE_DIR = Path("/n/home04/rrinberg/data_dir__holylabs/unlearning/precomputed_models/oracles/CIFAR10/")
MIT_ORACLE_DIR = Path("/mnt/xfs/projects/untrak/MATCHING/oracles/CIFAR10/")
if HARVARD_ORACLE_DIR.exists():
    ORACLE_BASE_DIR = HARVARD_ORACLE_DIR
elif MIT_ORACLE_DIR.exists():
    ORACLE_BASE_DIR = MIT_ORACLE_DIR
else:
    raise ValueError("No oracle directory found")

N_models_for_direct = 100
N_models_for_direct = 100



def do_baselines():
    import sys
    from itertools import product
    index = int(sys.argv[1])

    forget_set_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    #num_epochs_groups = [5, 10, 15, 20]
    method_names = ["do_nothing", "load_an_oracle"]
    num_epochs = 10

    tups = list(product(forget_set_ids, method_names))
    print(f"tups are :{tups}")
    tup = tups[index]
    forget_set_id, method_name = tup

    CWD = Path.cwd()
    BASE_DIR = CWD.parent.parent

    config_file = BASE_DIR / "configs" / "test_oracle_matching.yaml"

    #config_file.exists()
    ###o
    config_dict = read_yaml(config_file)
    #config_dict["results_dir"] = "./scrub_results/"
    config_dict["forget_set_id"] = forget_set_id
    config_dict["unlearning_algo_kwargs"]["oracles_path"] = ORACLE_BASE_DIR/ f"forget_set_{forget_set_id}"
    
    config_dict["unlearning_algo_kwargs"]["forget_set_id"] = forget_set_id

    config_dict["run_direct_eval"] = True
    config_dict["unlearning_algo_kwargs"]["num_epochs"] = num_epochs
    config_dict["unlearning_algo"] = method_name
    config_dict["N_models_for_direct"] = N_models_for_direct
    config_dict["unlearning_algo_kwargs"][
        "N_models_for_direct"] = N_models_for_direct
    

    config_dict["results_dir"] = str(RESULTS_DIR)

    print(f"running config_dict ")
    for k, v in config_dict.items():
        print(f"{k} - {v}")
        
    results_dir, results = eval_suite.eval_suite(config_dict=config_dict)


if __name__ == "__main__":
    do_baselines()

