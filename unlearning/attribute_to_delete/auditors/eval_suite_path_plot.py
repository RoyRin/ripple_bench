import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
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
from unlearning.datasets import DATASET_SIZES, DATASET_VAL_SIZES
from unlearning.unlearning_algos.base_nn import NAME_TO_ALGO


def read_yaml(yaml_file):
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_model(path, model_factory, ds_name):
    model = model_factory(ds_name)
    loaded_model = ch.load(path)
    if ds_name == "QNLI":
        model.model.load_state_dict(loaded_model)
        return model

    first_key = list(loaded_model.keys())[0]
    print("> first_key:", first_key)
    if "model" in first_key:
        model.load_state_dict(loaded_model)
    else:
        # add ".model" to each key in k,vs
        loaded_model = {f"model.{k}": v for k, v in loaded_model.items()}
        model.load_state_dict(loaded_model)
    return model


def _setup(config_yaml_file=None, config_dict=None, overwrite=False, unlearn_fn=None):
    ####### SETUP ########
    if config_yaml_file is not None:
        config = read_yaml(config_yaml_file)
    elif config_dict is not None:
        config = config_dict
    else:
        raise ValueError("Must pass in either a yaml file or a dictionary")

    results = {}
    results["params"] = {}

    pprint.pp(config)
    RESULTS_DIR = make_results_dir(config)
    # save config to results dir

    with open(RESULTS_DIR / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Overwrite check (checks only for `direct`, but we can change this if we want)
    direct_results_file = RESULTS_DIR / "path_plot_klom.npy"
    if not overwrite and direct_results_file.exists():
        raise FileExistsError(
            f"{RESULTS_DIR} already exists. Set overwrite=True to overwrite."
        )

    logger = logging.getLogger("EvalSuite")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.CRITICAL,  # ignore submitit warnings
        handlers=[
            logging.FileHandler(RESULTS_DIR / "eval_suite.log"),
            logging.StreamHandler(),
        ],
    )
    logger.setLevel(logging.INFO)
    logger.info("Setting up eval suite..")
    rng = np.random.RandomState(0)

    ds_name = config["dataset"]

    # for now, let's tie the model to the dataset, so we have fewer moving pieces
    model = model_factory(ds_name)  # on cuda, in eval mode
    logger.info(f"Loaded model.")

    forget_set_indices = load_forget_set_indices(ds_name, config["forget_set_id"])

    results["params"]["forget_set_indices"] = forget_set_indices
    if unlearn_fn is None:
        unlearn_fn = NAME_TO_ALGO[config["unlearning_algo"]]

    unlearning_kwargs = config["unlearning_algo_kwargs"]
    if unlearning_kwargs is None:
        unlearning_kwargs = {}
    logger.info(f"Loaded unlearning algo: {config['unlearning_algo']}")

    with redirect_stdout(open("/dev/null", "w")):
        # no shuffling, no augmentation
        if ds_name == "QNLI":
            train_indices = np.arange(10_000)
        elif ds_name == "LIVING17":
            train_indices = np.arange(44_200)
        else:
            train_indices = np.arange(50_000)

        train_loader = loader_factory(ds_name, indices=train_indices, indexed=True)
        val_loader = loader_factory(ds_name, split="val", indexed=True)
        forget_loader = loader_factory(
            ds_name,
            indices=forget_set_indices,
            batch_size=50,
            indexed=True,
        )

        eval_set_inds = np.arange(DATASET_SIZES[ds_name] + DATASET_VAL_SIZES[ds_name])
        eval_loader = loader_factory(
            ds_name, split="train_and_val", indices=eval_set_inds, indexed=True
        )

    logger.info(f"Created loaders.")

    return (
        config,
        ds_name,
        unlearn_fn,
        logger,
        rng,
        forget_set_indices,
        eval_set_inds,
        train_loader,
        forget_loader,
        eval_loader,
        unlearning_kwargs,
        results,
        RESULTS_DIR,
    )


def get_path_plot(
    config_yaml_file=None, config_dict=None, overwrite=False, unlearn_fn=None
):
    """
    pass in either a yaml file or a dictionary

    expects that unlearn_fn has the interface:
        unlearned_model = unlearn_fn(model,
                                     train_dataloader,
                                     forget_dataloader,
                                     forget_indices,
                                     **kwargs)
    """
    (
        config,
        ds_name,
        unlearn_fn,
        logger,
        rng,
        forget_set_indices,
        eval_set_inds,
        train_loader,
        forget_loader,
        eval_loader,
        unlearning_kwargs,
        results,
        RESULTS_DIR,
    ) = _setup(config_yaml_file, config_dict, overwrite, unlearn_fn)

    ####### LOAD PRETRAINED MODELS ########
    # inserted by Roy for some speed reason
    splits = ["train", "val"]

    f_ckpt_paths, f_logit_paths, f_margins_paths = get_full_model_paths(
        ds_name, splits=splits
    )

    (
        o_ckpt_0_path,  # we only need a single oracle checkpoint
        o_logit_paths,
        o_margins_paths,
    ) = get_oracle_paths(ds_name, config["forget_set_id"], splits=splits)
    logger.info(f"Loaded paths of pretrained models.")

    model = load_model(f_ckpt_paths[0], model_factory, ds_name)

    _inds_to_plot = rng.choice(np.arange(len(forget_set_indices)), 20)
    inds_to_plot = forget_set_indices[_inds_to_plot]

    logger.info(f"Loaded a pretrained model.")
    ####### END OF LOADING PRETRAINED MODELS ########

    ####### RUN UNLEARNING ALGO ONCE ########
    # run the unlearning algo on the model we just created once
    logger.info(f"Running unlearning algo..")

    unlearned_model = unlearn_fn(
        model=model,
        train_dataloader=train_loader,
        forget_dataloader=forget_loader,
        forget_indices=forget_set_indices,
        **unlearning_kwargs,
    )
    logger.info(f"Done running unlearning algo.")
    ####### END OF RUNNING UNLEARNING ALGO ########
    return RESULTS_DIR, results
