from unlearning.datasets.cifar10 import (
    get_dataloader,
    get_cifar_dataset,
    get_cifar_dataloader,
)
from unlearning.models.resnet9 import ResNet9
from unlearning.training.train import train_cifar10, eval_cifar10
from unlearning.unlearning_algos.utils import get_margin
from pathlib import Path
from tqdm import tqdm
import torch as ch
import numpy as np


def _margin_score(model: ch.nn.Module, batch: list[ch.Tensor]) -> ch.Tensor:
    x, y, inds = batch
    with ch.no_grad():
        res = get_margin(model, x, y, inds).clone().detach().cpu()
    return res


def _grad_norm_score(model: ch.nn.Module, batch: list[ch.Tensor]) -> ch.Tensor:
    x, y, index = batch
    outs = model(x, index)[ch.arange(x.shape[0]), y]
    ws = []
    for out in outs:
        g = ch.autograd.grad(out, model.parameters(), retain_graph=True)
        g = ch.tensor([ch.norm(gi, p=ch.inf) for gi in g]).max().item()
        ws.append(g)
    return ch.tensor(ws).clone().detach().cpu()


def _get_epsilon(scores: ch.Tensor) -> float:
    r = scores.shape[0]
    plus_inds = ch.arange(r // 2).tolist()
    minus_inds = ch.arange(r // 2, r).tolist()

    sorted_scores = ch.argsort(scores, descending=True)
    k_plus = set(sorted_scores[: r // 2].tolist())
    k_minus = set(sorted_scores[r // 2 :].tolist())
    correct_plus = k_plus.intersection(set(plus_inds))
    correct_minus = k_minus.intersection(set(minus_inds))
    num_correct_guesses = len(correct_plus) + len(correct_minus)
    print(f"num_correct_guesses={num_correct_guesses}")

    beta = 0.05
    osqrtr = np.sqrt(np.log(1 / beta) * r / 2)
    for epsilon in ch.arange(0, 10, 0.1):
        v = osqrtr + r * np.exp(epsilon) / (1 + np.exp(epsilon))
        print(f"At epsilon={epsilon}, v={v}")
        if num_correct_guesses > v:
            print(f"Rejected epsilon={epsilon}")
        else:
            print(f"Failed to reject epsilon={epsilon}!")
            return epsilon
    # if we never reject, return the largest epsilon
    print("Failed to reject any epsilon!")
    return epsilon


def audit_o1(
    model: ch.nn.Module,
    dataset: ch.utils.data.Dataset,
    score_fn: callable,
    trainer_fn: callable,
    S_plus: np.ndarray,
    S_minus: np.ndarray,
    S_rest: np.ndarray,
    CKPT_DIR=Path("/mnt/xfs/home/krisgrg/projects/unlearning-with-trak/data/o1_ckpts"),
    overwrite_ckpts: bool = False,
    N_models: int = 10,
    **kwargs,
) -> float:
    """
    |S_plus + S_minus| = m
    |S_plus| = |S_minus| = m // 2
    |S_rest| = N - m
    """
    S_train = np.concatenate([S_plus, S_rest])
    train_dataloader = get_dataloader(dataset, indices=S_train)
    print(len(train_dataloader.dataset))

    S = []

    for i in range(N_models):
        model = ResNet9(num_classes=10).cuda()
        model = trainer_fn(
            model,
            train_dataloader,
            checkpoints_dir=CKPT_DIR,
            model_save_suffix=f"model_{i}",
            overwrite=overwrite_ckpts,
        )

        model = model.eval()

        ms = np.concatenate([S_plus, S_minus])
        m_dataloader = get_dataloader(dataset, indices=ms)

        scores = []
        for x, y in tqdm(m_dataloader, desc="computing scores..."):
            batch = [x.cuda(), y.cuda()]
            score = score_fn(model, batch)
            scores.append(score)
        scores = ch.cat(scores)
        S.append(scores)
    S = ch.stack(S).mean(dim=0)
    epsilon = _get_epsilon(S)
    return epsilon, S


def audit_o1_cifar(
    model: ch.nn.Module,
    score_fn: callable,
    S_plus: np.ndarray,
    S_minus: np.ndarray,
    S_rest: np.ndarray,
    CKPT_DIR=Path("/mnt/xfs/home/krisgrg/projects/unlearning-with-trak/data/o1_ckpts"),
    overwrite_ckpts: bool = False,
    N_models: int = 10,
) -> float:
    cifar_dataset = get_cifar_dataset(split="train", augment=True)

    return audit_o1(
        model=model,
        dataset=cifar_dataset,
        score_fn=score_fn,
        trainer_fn=train_cifar10,
        S_plus=S_plus,
        S_minus=S_minus,
        S_rest=S_rest,
        CKPT_DIR=CKPT_DIR,
        overwrite_ckpts=overwrite_ckpts,
        N_models=N_models,
    )


def audit_o1_external_data(
    model: ch.nn.Module,
    score_fn: callable,
    trainer_fn: callable,
    ds_plus: ch.utils.data.Dataset,
    ds_minus: ch.utils.data.Dataset,
    ds_rest: ch.utils.data.Dataset,
    CKPT_DIR=Path("/mnt/xfs/home/krisgrg/projects/unlearning-with-trak/data/o1_ckpts"),
    overwrite_ckpts: bool = False,
    **kwargs,
) -> float:
    """
    ds_plus: dataset of the positive examples
    ds_minus: dataset of the negative examples
    ds_rest: dataset of the rest of the examples
    """
    # TODO: translate datasets to indices of a merged dataset and call audit_o1

    pass
