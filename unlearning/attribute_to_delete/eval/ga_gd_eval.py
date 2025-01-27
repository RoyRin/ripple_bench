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


# Parameters selected based on the lowest forget set CE on forget set 3
# see `scrap.ipynb`
best_params_GD = {
    "method_name": "benchmark_GD_wrapper",
    "unlearning_algo": "unlearning_algo",
    "num_epochs": 5,
    "learning_rate": 0.001,
    "forget_batch_size": 64,
}

best_params_GA = {
    "method_name": "benchmark_GA_wrapper",
    "unlearning_algo": "benchmark_GA_wrapper",
    "num_epochs": 5,
    "learning_rate": 0.001,
    "batch_size": 64,
    "forget_batch_size": 64,
}



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


def do_GA_GD():

    index = int(sys.argv[1])
    
    forget_batch_size = 64
    batch_size = 64

    forget_set_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    method_names = ["benchmark_GD_wrapper", "benchmark_GA_wrapper"]
    learning_rates = [1e-5, 1e-3, 1e-2]

    num_epochs_group = [2, 5, 7, 10]
    num_epochs_group = [1, 3, 5, 7, 10]
    # 9 * 2 * 3 = 54
    # 54 *4 = 216

    # 9 *2 *3 * 5 = 270


    tups = list(product(forget_set_ids, method_names, num_epochs_group, learning_rates,))

    for i, t in enumerate(tups):
        print(f"tup {i} is :{t}")
    tup = tups[index]

    print(f"{index} tup is :{tup}")

    
    forget_set_id, method_name, num_epochs, learning_rate = tup

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
    config_dict["N_models_for_direct"] = N_models_for_direct
    config_dict["unlearning_algo_kwargs"]["N_models_for_direct"] = N_models_for_direct

    config_dict["unlearning_algo_kwargs"]["num_epochs"] = num_epochs
    config_dict["unlearning_algo_kwargs"]["learning_rate"] = learning_rate
    config_dict["unlearning_algo_kwargs"]["batch_size"] = batch_size
    config_dict["unlearning_algo_kwargs"]["forget_batch_size"] = forget_batch_size


    config_dict["unlearning_algo"] = method_name

    config_dict["results_dir"] = str(RESULTS_DIR)
    
    print(f"running config_dict ")
    for k, v in config_dict.items():
        print(f"{k} - {v}")

    import datetime
    start = datetime.datetime.now()

    results_dir, results = eval_suite.eval_suite(config_dict=config_dict)
    end = datetime.datetime.now()
    print(f"Time taken: {end-start}")



if __name__ == "__main__":
    do_GA_GD()

