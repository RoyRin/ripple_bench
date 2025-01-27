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
from unlearning.auditors.ulira import (cheap_ulira_audit_precomputed, ulira_strong, get_ulira_forget_mask, get_ulira_model_paths,get_ulira_masks,  ulira_paper, load_all_ulira_margins, train_unlearnings)
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
    if ds_name == 'QNLI':
        model.model.load_state_dict(loaded_model)
        return model

    first_key = list(loaded_model.keys())[0]
    print('> first_key:', first_key)
    if "model" in first_key:
        model.load_state_dict(loaded_model)
    else:
        # add ".model" to each key in k,vs
        loaded_model = {f"model.{k}": v for k, v in loaded_model.items()}
        model.load_state_dict(loaded_model)
    return model


def eval_suite(config_yaml_file=None,
               config_dict=None,
               overwrite=False,
               unlearn_fn=None):
    """
    pass in either a yaml file or a dictionary

    expects that unlearn_fn has the interface:
        unlearned_model = unlearn_fn(model,
                                     train_dataloader,
                                     forget_dataloader,
                                     forget_indices,
                                     **kwargs)
    """
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
    direct_results_file = RESULTS_DIR / "direct" / "ks_p_kl_ce.npy"
    if not overwrite and direct_results_file.exists():
        raise FileExistsError(
            f"{RESULTS_DIR} already exists. Set overwrite=True to overwrite.")

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

    forget_set_indices = load_forget_set_indices(ds_name,
                                                 config["forget_set_id"])

    results["params"]["forget_set_indices"] = forget_set_indices
    if unlearn_fn is None:
        unlearn_fn = NAME_TO_ALGO[config["unlearning_algo"]]

    unlearning_kwargs = config["unlearning_algo_kwargs"]
    if unlearning_kwargs is None:
        unlearning_kwargs = {}
    logger.info(f"Loaded unlearning algo: {config['unlearning_algo']}")

    with redirect_stdout(open("/dev/null", "w")):
        # no shuffling, no augmentation
        if ds_name == 'QNLI':
            train_indices = np.arange(10_000)
        elif ds_name == 'LIVING17':
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
        eval_loader = loader_factory(ds_name,
                                     split="train_and_val",
                                     indices=eval_set_inds,
                                     indexed=True)

    logger.info(f"Created loaders.")
    ####### END OF SETUP ########

    ####### LOAD PRETRAINED MODELS ########
    # f stands for "full model",
    #   i.e. the model that was trained on the enitre dataset (retain + forget)
    # o stands for "oracle model"
    #   i.e. the model that was trained on the oracle dataset (retain only)

    # inserted by Roy for some speed reason
    splits = ["train", "val"]

    f_ckpt_paths, f_logit_paths, f_margins_paths = get_full_model_paths(
        ds_name, splits=splits)

    (
        o_ckpt_0_path,  # we only need a single oracle checkpoint
        o_logit_paths,
        o_margins_paths,
    ) = get_oracle_paths(ds_name, config["forget_set_id"], splits=splits)
    logger.info(f"Loaded paths of pretrained models.")

    #print('f_ckpt_path[0]:', f_ckpt_paths[0])
    model = load_model(f_ckpt_paths[0], model_factory, ds_name)

    _inds_to_plot = rng.choice(np.arange(len(forget_set_indices)), 20)
    inds_to_plot = forget_set_indices[_inds_to_plot]

    logger.info(f"Loaded a pretrained model.")
    ####### END OF LOADING PRETRAINED MODELS ########
    if not config.get("only_direct_eval", True):

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

        '''
        ####### SAVE & PLOT LOGITS ########
        LOGIT_EVAL_DIR = RESULTS_DIR / "logit_eval"
        LOGIT_EVAL_DIR.mkdir(parents=True, exist_ok=True)
        unlearned_logits = compute_logits(unlearned_model, eval_loader).numpy()
        np.save(LOGIT_EVAL_DIR / "unlearned_logits.npy", unlearned_logits)
        logger.info(
            f"Saved unlearned logits to {LOGIT_EVAL_DIR / 'unlearned_logits.npy'}"
        )

        fig = plot_logits(
            unlearned_logits=unlearned_logits,
            full_logit_paths=f_logit_paths,
            oracle_logit_paths=o_logit_paths,
            n_full_logits_to_plot=20,
            n_oracle_logits_to_plot=20,
            inds_to_plot=inds_to_plot,
            reorder_classes=config.get("reorder_logit_classes",
                                       False),  # keep for Roy
        )

        fig.savefig(LOGIT_EVAL_DIR / "logits_figure.png")
        logger.info(
            f"Saved logit figure to {LOGIT_EVAL_DIR / 'logits_figure.png'}")
        ####### END OF SAVING & PLOTTING LOGITS ########
        '''

        ####### EVALUATE ACCURACIES ########
        # eval the unlearned model accuracy on train/val/forget set
        train_acc, val_acc, forget_acc = eval_accuracy(
            unlearned_model,
            train_loader,
            val_loader,
            forget_loader,
        )
        oracle_model = load_model(o_ckpt_0_path, model_factory, ds_name)
        #print(oracle_model_path)

        oracle_model.cuda().eval()  # can't be too sure
        oracle_train_acc, oracle_val_acc, oracle_forget_acc = eval_accuracy(
            oracle_model, train_loader, val_loader, forget_loader)

        print("=" * 20)
        print(f"Train acc:  {train_acc:.2f} (oracle: {oracle_train_acc:.2f})")
        print(f"Val acc:    {val_acc:.2f} (oracle: {oracle_val_acc:.2f})")
        print(
            f"Forget acc: {forget_acc:.2f} (oracle: {oracle_forget_acc:.2f})")
        print("=" * 20)
        # save csv with the accuracies
        # columns are forget_set_id, unlearning_algo, train_acc, val_acc, forget_acc
        # and there are two rows, one with the unlearned model, one with the oracle
        evaluations = {
            "forget_set_id":
            [config["forget_set_id"], config["forget_set_id"]],
            "unlearning_algo": [config["unlearning_algo"], "oracle"],
            "train_acc": [train_acc, oracle_train_acc],
            "val_acc": [val_acc, oracle_val_acc],
            "forget_acc": [forget_acc, oracle_forget_acc],
        }
        df = pd.DataFrame(evaluations)
        df.to_csv(RESULTS_DIR / "accuracies.csv")
        logger.info(f"Saved accuracies to {RESULTS_DIR / 'accuracies.csv'}")
        results.update(evaluations)

        ####### END OF EVALUATING ACCURACIES ########
        print(f"size of forget_set_indices - {forget_set_indices.shape}")

        '''
        ####### (NAIVE) LIRA EVAL ########
        LIRA_DIR = RESULTS_DIR / "lira"
        LIRA_DIR.mkdir(parents=True, exist_ok=True)
        (
            lira_ratios,
            all_unlearned_margins,
            all_full_margins,
            all_oracle_margins__quick,
        ) = cheap_ulira_audit_precomputed(
            dataset_name=ds_name,
            loader_factory=loader_factory,
            unlearned_model=unlearned_model,
            full_margins_paths=f_margins_paths,
            oracle_margins_paths=o_margins_paths,
            forget_set_indices=forget_set_indices,
            val_set_size=len(val_loader.dataset),
        )

        logger.info(f"Done running LiRA eval..")
        cheap_ulira_results = {
            "lira_ratios": lira_ratios,
            "all_unlearned_margins": all_unlearned_margins,
            "all_full_margins": all_full_margins,
            "all_oracle_margins__quick": all_oracle_margins__quick,
        }
        results["cheap_ulira"] = cheap_ulira_results

        np.save(LIRA_DIR / "lira_ratios.npy", lira_ratios.numpy())
        ratios_fig, ratios_ax = plt.subplots()
        sns.histplot(
            lira_ratios.clamp(min=0, max=10).numpy(),
            stat="probability",
            kde=True,
            ax=ratios_ax,
        )
        ratios_ax.set_xlabel(
            "LiRA ratio: P(unlearned ~ full) / P(unlearned ~ oracle); clipped at 10"
        )
        ratios_fig.savefig(LIRA_DIR / "lira_ratios.png")
        np.save(LIRA_DIR / "lira_unlearned_margins.npy",
                all_unlearned_margins.numpy())
        logger.info(
            f"Saved LiRA unlearned margins to {LIRA_DIR / 'lira_unlearned_margins.npy'}"
        )

        fig_margins = plot_margins(
            all_unlearned_margins,
            all_full_margins,
            all_oracle_margins__quick,
            inds_to_plot=_inds_to_plot,
        )
        fig_margins.savefig(LIRA_DIR / "margins_hist.png")
        logger.info(f"Saved LiRA results in {LIRA_DIR}")
        ####### END OF LIRA EVAL ########
        '''

    ####### DIRECT EVAL ########
    if config.get("run_direct_eval", False):


        DIRECT_DIR = RESULTS_DIR / "direct"
        DIRECT_DIR.mkdir(parents=True, exist_ok=True)
        # run the expensive direct eval
        # this involves running the unlearning algo many times
        # (once on each checkpoint from full_model_ckpt_paths)
        all_unlearned_margins = []
        if config["use_submitit_for_direct_eval"]:
            executor = config_submitit(config, RESULTS_DIR)
            executor.update_parameters(slurm_partition="background")

            N_jobs = min(config["N_models_for_direct"], len(f_ckpt_paths))
            logger.info(f"Running direct eval with {N_jobs} jobs..")

            Path(executor.folder).mkdir(parents=True, exist_ok=True)

            with redirect_stderr(open(Path(executor.folder) / "log.txt", "w")):
                job_array = executor.map_array(
                    u_margin_job,
                    f_ckpt_paths,
                    [unlearn_fn] * N_jobs,
                    [unlearning_kwargs] * N_jobs,
                    [ds_name] * N_jobs,
                    [model_factory] * N_jobs,
                    [loader_factory] * N_jobs,
                    [forget_set_indices] * N_jobs,
                    [eval_set_inds] * N_jobs,
                )
            logger.info(f"Sent {N_jobs} jobs..")
            num_failed_jobs = 0
            for job in job_array:
                try:
                    all_unlearned_margins.append(job.result())
                except Exception as e:
                    num_failed_jobs += 1
                    print(e)
            try:
                all_unlearned_margins = ch.stack(all_unlearned_margins)
            except Exception as e:
                print(e)
            logger.info(
                f"Successfully computed {len(all_unlearned_margins)} unlearned margins"
            )
            if num_failed_jobs == 0:
                if os.path.exists(executor.folder):
                    shutil.rmtree(executor.folder)
                    logger.info(
                        f"Deleted submitit logs directory {executor.folder} and all files inside."
                    )
        else:  # no submitit, locally run the jobs sequentially

            f_ckpt_paths = f_ckpt_paths[:config["N_models_for_direct"]]
            for f_ckpt_path in tqdm(f_ckpt_paths,
                                    desc="unlearning models.."):
                _m = get_u_margins(
                    model,
                    f_ckpt_path,
                    unlearn_fn,
                    unlearning_kwargs,
                    train_loader,
                    eval_loader,
                    forget_set_indices,
                )
                all_unlearned_margins.append(_m)
            all_unlearned_margins = ch.stack(all_unlearned_margins)

        logger.info(f"Done computing margins for direct eval..")
        if config.get("save_unlearned_margins", True):
            np.save(
                DIRECT_DIR / "direct_unlearned_margins.npy",
                all_unlearned_margins.numpy(),
            )

        logger.info("Loading oracle margins..")
        assert len(o_margins_paths) == 2
        assert o_margins_paths[0].stem.startswith("train")
        assert o_margins_paths[1].stem.startswith("val")
        all_oracle_margins = ch.cat(
            [ch.load(path) for path in o_margins_paths], dim=1)
        all_oracle_margins = all_oracle_margins[:, eval_set_inds]

        fig_margins_direct = plot_margins_direct(all_unlearned_margins,
                                                 all_oracle_margins,
                                                 inds_to_plot)
        fig_margins_direct.savefig(DIRECT_DIR / "direct_margins_hist.png")

        direct_results = direct_audit_precomputed(all_unlearned_margins,
                                                  all_oracle_margins)

        print(direct_results)
        np.save(DIRECT_DIR / "ks_p_kl_ce.npy", direct_results)

        if False:
            ulira_strong_results = ulira_strong(all_unlearned_margins,
                                            all_oracle_margins)

        direct_ulira_results = {
            "all_unlearned_margins": all_unlearned_margins,
            "direct_results": direct_results,
            #"ulira_strong": ulira_strong_results,
        }
        results["direct_ulira"] = direct_ulira_results

        fig_direct_results, dir_axes = plt.subplots(1, 3)
        for i, col in enumerate(["ks pval", "t pval", "KL divergence"]):
            print(f"direct_results.shape - {direct_results.shape}")
            dir_axes[i].hist(direct_results[:, i], bins=20)
            dir_axes[i].set_title(col)
        fig_direct_results.savefig(DIRECT_DIR / "direct_results_hist.png")
        # save dictionary direct_results as npz
        np.savez(DIRECT_DIR / "direct_results.npz", **direct_ulira_results)

    return RESULTS_DIR, results


if __name__ == "__main__":

    FID = 3

    '''
    config = {
        # subdir with rest of hparams will be created for you
        "results_dir": "/mnt/xfs/projects/untrak/RESULTS_L17",
        "dataset": "LIVING17",
        "forget_set_id": FID,
        "unlearning_algo": "load_an_oracle",
        "unlearning_algo_kwargs":
        {
            "dataset": "LIVING17",
            "forget_set_id": FID,
        },
        "run_direct_eval": True,
        "only_direct_eval": False,
        "use_submitit_for_direct_eval": True,
        "N_models_for_direct": 100,
    }
    '''

    config = {
        "results_dir": "/mnt/xfs/projects/untrak/RESULTS_L17",
        "dataset": "LIVING17",
        "forget_set_id": FID,
        "unlearning_algo": "dm_direct",
        "unlearning_algo_kwargs":
        {
            "dm_scores_path": "/mnt/xfs/projects/untrak/MATCHING/DATAMODELS/living17/dms/all_logit_dms.npy",
            "dataset": "LIVING17",
            "multiplier": 10.0,
        },
        "run_direct_eval": True,
        "only_direct_eval": False,
        "use_submitit_for_direct_eval": False,
        "N_models_for_direct": 100,
    }


    #RESULTS_DIR = eval_suite(config_dict=config, overwrite=True)
    config = {
            # subdir with rest of hparams will be created for you
            "results_dir": "/mnt/xfs/projects/untrak/RESULTS_L17",
            "dataset": "LIVING17",
            "forget_set_id": FID,
            "unlearning_algo": "oracle_matching",
            "unlearning_algo_kwargs":
            {
                "dataset": "LIVING17",
                "oracles_path": f"/mnt/xfs/projects/untrak/MATCHING/oracles/LIVING17/forget_set_{FID}/",
                "loss_type": "MSE",
                "num_epochs": 2,
                "learning_rate": 1e-4,
                "batch_size": 512,
                "optimizer": "adam",
                "retain_multiplier": 20.,
                "forget_multiplier": 1.,
                "num_oracles": 1,
                "wd_lambda": 0.,
                "shuffle": True,
            },
            "run_direct_eval": True,
            "only_direct_eval": False,
            "use_submitit_for_direct_eval": False,
            "N_models_for_direct": 50,
        }

    RESULTS_DIR = eval_suite(config_dict=config, overwrite=True)

    '''
    config = {
        # subdir with rest of hparams will be created for you
        "results_dir": "/mnt/xfs/projects/untrak/RESULTS_L17",
        "dataset": "LIVING17",
        "forget_set_id": FID,
        "unlearning_algo": "do_nothing",
        "unlearning_algo_kwargs":
        {
        },
        "run_direct_eval": True,
        "only_direct_eval": False,
        "use_submitit_for_direct_eval": True,
        "N_models_for_direct": 100,
    }

    RESULTS_DIR = eval_suite(config_dict=config, overwrite=True)
    #RESULTS_DIR = eval_suite(config_dict=config, overwrite=True)
    '''

    '''
    FID = 3
    config = {
            # subdir with rest of hparams will be created for you
            "results_dir": "/mnt/xfs/projects/untrak/QNLI_TEST",
            "dataset": "QNLI",
            "forget_set_id": FID,
            "unlearning_algo": "do_nothing",
            "unlearning_algo_kwargs":
            {
            },
            "run_direct_eval": True,
            "only_direct_eval": False,
            "use_submitit_for_direct_eval": True,
            "N_models_for_direct": 100,
        }

    '''

    '''
    FID = 4
    config = {
            # subdir with rest of hparams will be created for you
            "results_dir": "/mnt/xfs/projects/untrak/QNLI_TEST",
            "dataset": "QNLI",
            "forget_set_id": FID,
            "unlearning_algo": "oracle_matching",
            "unlearning_algo_kwargs":
            {
                "dataset": "QNLI",
                "oracles_path": f"/mnt/xfs/projects/untrak/MATCHING/oracles/QNLI/forget_set_{FID}/",
                "loss_type": "MSE",
                "num_epochs": 0,
                "learning_rate": 1e-4,
                "batch_size": 512,
                "optimizer": "adam",
                "retain_multiplier": 5.,
                "forget_multiplier": 1.,
                "num_oracles": 1,
                "wd_lambda": 0.,
                "shuffle": True,
            },
            "run_direct_eval": True,
            "only_direct_eval": False,
            "use_submitit_for_direct_eval": False,
            "N_models_for_direct": 100,
        }

    RESULTS_DIR = eval_suite(config_dict=config, overwrite=True)
    '''


    '''
    import sys
    from itertools import product

    try:
        INDEX = int(sys.argv[1])
    except Exception as e:
        print(f"e - {e}")
        INDEX = 0

    algo_names = [
        "do_nothing",
        "load_an_oracle",
        # "logit_trak",
        "oracle_matching",
        # "gradient_ascent",
        # "gradient_ascent_grid_search",
        "logit_trak__addendum",
        "benchmark_GD_wrapper",
        "benchmark_GA_wrapper",
    ]
    # 8 * 3
    forget_sets_of_interest = [1, 5, 9]
    tups = list(product(forget_sets_of_interest, algo_names))

    tup = tups[INDEX]
    (forget_set_id, unlearning_algo_name) = tup

    default_config_dir = Path("/n/home04/rrinberg/code/unlearning-with-trak/configs")
    default_config_path = default_config_dir / "roy_default_config.yaml"

    config = read_yaml(default_config_path)

    config["forget_set_id"] = forget_set_id
    config["unlearning_algo"] = unlearning_algo_name

    print(f"{INDEX} -Running {forget_set_id} {unlearning_algo_name}")
    eval_suite(config_yaml_file=None, config_dict=config)
    print(f"all done !")
    '''


def plot_single_histogram(
    index,
    forget_set_id,
    RESULT_DIR,
    ORACLES_BASE_DIR=Path("/mnt/xfs/projects/untrak/MATCHING/oracles/CIFAR10"),
):
    unlearned_margins = np.load(RESULT_DIR / "direct/direct_unlearned_margins.npy")
    train_om = ch.load(
        ORACLES_BASE_DIR / f"forget_set_{forget_set_id}/train_margins_all.pt"
    )
    val_om = ch.load(
        ORACLES_BASE_DIR / f"forget_set_{forget_set_id}/val_margins_all.pt"
    )
    oracle_margins = ch.cat([train_om, val_om], dim=1).numpy()
    bins = np.linspace(
        min(np.min(unlearned_margins[:, index]), np.min(oracle_margins[:, index])),
        max(np.max(unlearned_margins[:, index]), np.max(oracle_margins[:, index])),
        30,
    )
    # sns.histplot(
    #     unlearned_margins[:, index],
    #     alpha=0.8,
    #     stat="probability",
    #     bins=bins,
    #     label=r"$\hat f$",
    #     color="#753180",
    # )
    sns.histplot(
        oracle_margins[:, index],
        alpha=0.8,
        stat="probability",
        # bins=bins,
        bins=20,
        label=r"$f$",
        color="grey",
    )
    mean = oracle_margins[:, index].mean()
    plt.axvline(mean, color="#753180", linestyle="--", linewidth=4)
    plt.legend(prop={"size": 20})
    plt.show()
