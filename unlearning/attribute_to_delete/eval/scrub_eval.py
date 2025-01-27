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



best_params_SCRUB = {
    "method_name": "scrub",
    "unlearning_algo": "scrub",
    "num_epochs": 10,
    "learning_rate": 0.01,
    "forget_batch_size": 32,
    "beta": 0.999,
    "retain_batch_size": 64,
    "forget_batch_size": 32,
    "maximization_epochs": 3
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


def do_scrub_eval():
    index = int(sys.argv[1])
    beta = 0.9999  # 0.95, 0.999
    retain_bs = 64
    forget_bs = 32  # 64

    maximization_epochs = [3, 5]
    #num_epochs = [3, 7, 10]

    num_epochs_group = [1, 3, 5, 7, 10]

    forget_set_ids = [1,2, 3, 4,5, 6,7,8,9]
    ### 9 * 3 * 2 = 54
    # 9 * 2 * 5 = 90

    tups = list( product(forget_set_ids, maximization_epochs, num_epochs_group))
    for i, t in enumerate(tups):
        print(f"tup {i} is :{t}")
    tup = tups[index]
    print(f"tup is :{tup}")
    forget_set_id, max_epochs, num_epochs = tup

    print(f"forget_set_id, beta, retain_b, forget_b, max_epochs, num_epochs ")
    print(
        f"{forget_set_id}, {beta}, {retain_bs}, {forget_bs}, {max_epochs}, {num_epochs}"
    )
    #tups = list(product(forget_set_ids, num_epochs_groups))

    CWD = Path.cwd()
    BASE_DIR = CWD.parent.parent

    config_file = BASE_DIR / "configs" / "test_oracle_matching.yaml"

    config_file.exists()

    config_dict = read_yaml(config_file)

    config_dict["forget_set_id"] = forget_set_id
    config_dict["unlearning_algo_kwargs"][
        "oracles_path"] = f"/n/home04/rrinberg/data_dir__holylabs/unlearning/precomputed_models/oracles/CIFAR10/forget_set_{forget_set_id}"
    config_dict["unlearning_algo_kwargs"]["forget_set_id"] = forget_set_id

    config_dict["run_direct_eval"] = True
    config_dict["N_models_for_direct"] = N_models_for_direct
    config_dict["unlearning_algo"] = "scrub"

    config_dict["unlearning_algo_kwargs"]["num_epochs"] = num_epochs

    learning_rate = 0.01

    config_dict["unlearning_algo_kwargs"]["learning_rate"] = learning_rate
    config_dict["unlearning_algo_kwargs"]["beta"] = beta
    config_dict["unlearning_algo_kwargs"]["retain_batch_size"] = retain_bs
    config_dict["unlearning_algo_kwargs"]["forget_batch_size"] = forget_bs
    config_dict["unlearning_algo_kwargs"]["maximization_epochs"] = max_epochs
    config_dict["unlearning_algo_kwargs"]["N_models_for_direct"] = N_models_for_direct

    config_dict["results_dir"] = str(RESULTS_DIR)
    
    print(f"running config_dict ")
    for k, v in config_dict.items():
        print(f"{k} - {v}")

    results_dir, results = eval_suite.eval_suite(config_dict=config_dict)




if __name__ == "__main__":
    do_scrub_eval()
    