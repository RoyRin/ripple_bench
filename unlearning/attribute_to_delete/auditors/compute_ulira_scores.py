import re
import numpy as np
import torch as ch
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional

from unlearning.unlearning_algos.utils import get_margin
from unlearning.auditors.direct import (
    direct_audit_precomputed,
    get_u_margins,
    u_margin_job,
    plot_margins_direct,
)
from scipy.stats import norm
import torch
from unlearning.auditors.utils import (
    loader_factory,
 )


import os
import shutil
import yaml
import numpy as np
import torch as ch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import logging
import pprint
from contextlib import redirect_stdout, redirect_stderr

from unlearning.auditors.utils import (
    model_factory,
    loader_factory,
    load_forget_set_indices,
    get_full_model_paths,

    get_oracle_paths,
    make_results_dir,
)
from unlearning.auditors.accuracies import eval_accuracy
from unlearning.auditors.logit_plots import compute_logits, plot_logits
from unlearning.auditors.basic import plot_margins

from unlearning.auditors.direct import (
    config_submitit,
    direct_audit_precomputed,
    get_u_margins,
    u_margin_job,
    plot_margins_direct,
)
from unlearning.auditors import ulira
from unlearning.unlearning_algos.base_nn import NAME_TO_ALGO


from contextlib import redirect_stdout, redirect_stderr

ULIRA_BASE_DIR = Path(
    "/n/home04/rrinberg/data_dir__holylabs/unlearning/precomputed_models/ULIRA/"
)
if not ULIRA_BASE_DIR.exists():
    ULIRA_BASE_DIR = Path("/mnt/xfs/projects/untrak/MATCHING/ULIRA/")

def read_yaml(yaml_file):
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_model(path, model_factory, ds_name):
    model = model_factory(ds_name)
    loaded_model = ch.load(path)
    first_key = list(loaded_model.keys())[0]
    if "model" in first_key:
        model.load_state_dict(loaded_model)

    else:
        # add ".model" to each key in k,vs
        loaded_model = {f"model.{k}": v for k, v in loaded_model.items()}
        model.load_state_dict(loaded_model)
    return model


def get_all_margins():
    pass


if __name__ == "__main__":
    ds_name = "CIFAR10"
    training_masks = ulira.get_ulira_training_masks()
    forget_masks = ulira.get_ulira_forget_mask(ds_name, class_5_range=1000, overwrite=False )
    
    # which points to forget 
    ulira_train_all_margins, ulira_val_all_margins = ulira.load_all_ulira_margins(
        ds_name)

    all_oracle_margins_ulira = ch.cat([
        ch.tensor(ulira_train_all_margins.T),
        ch.tensor(ulira_val_all_margins.T)
    ]).T

    forget_indices = set([])
    for row in forget_masks:
        l = set(row.nonzero()[0])
        forget_indices= forget_indices.union(l)
    # this value should be 1000
    if len(forget_indices) != 1000:
        raise ValueError("forget_points should have 1000 points")
    # for each of these points, compute a ulira score using the margins
    for forget_index in forget_indices:
        pass 
        # get good margins
        # get bad margins
        all_unlearned_margins_ulira= None
        # TODO - all_unlearned_margins_ulira should be the margins that forget the models 
        ulira.ulira_paper(
            all_unlearned_margins_ulira,
            all_oracle_margins_ulira=all_oracle_margins_ulira,
            dataset_name="CIFAR10",
        )


# steps :
# 1. aggregate margins into a single collection of margins
# 2. compute which 1000 points we want to use for ULIRA
