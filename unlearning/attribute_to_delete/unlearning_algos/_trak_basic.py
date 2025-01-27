from typing import List, Callable
from trak.gradient_computers import FunctionalGradientComputer
from trak.modelout_functions import ImageClassificationModelOutput
from trak.projectors import CudaProjector, ProjectionType
from trak.utils import vectorize
import torch as ch
from copy import deepcopy

from .utils import add_vector_to_parameters


def trak_no_projection(
    model: ch.nn.Module,
    xtx_inv_matrix: ch.Tensor,
    lr: float = 1.0,
    train_dataloader: ch.utils.data.DataLoader = None,
    forget_indices: List[int] = None,
    forget_dataloader: ch.utils.data.Subset = None,
) -> ch.nn.Module:
    XTX_INV = xtx_inv_matrix.cuda()
    try:
        model_to_featurize = deepcopy(model)
        unlearned_model = deepcopy(model)
    except NotImplementedError:
        # MiniResNet is annoying to deepcopy
        model_to_featurize = model.custom_deepcopy()
        unlearned_model = model.custom_deepcopy()

    model_to_featurize = model_to_featurize.cuda()
    model_to_featurize.eval()

    unlearned_model = unlearned_model.cuda()
    unlearned_model.eval()

    grad_dim = sum([p.numel() for p in model_to_featurize.parameters()])
    if forget_dataloader is None:
        forget_ds = ch.utils.data.Subset(train_dataloader.dataset, forget_indices)
        forget_dataloader = ch.utils.data.DataLoader(
            forget_ds, batch_size=train_dataloader.batch_size
        )

    gradient_computer = FunctionalGradientComputer(
        model_to_featurize,
        ImageClassificationModelOutput(),
        grad_dim=grad_dim,
        dtype=ch.float32,
        device=ch.device("cuda"),
    )
    gradient_computer.load_model_params(model_to_featurize)

    delta = ch.zeros(1, grad_dim, device=ch.device("cuda"))
    olg = []
    for images, labels in forget_dataloader:
        batch = [images.cuda(), labels.cuda()]
        out_to_loss_grad = gradient_computer.compute_loss_grad(batch)

        grads = gradient_computer.compute_per_sample_grad(batch)
        grads = vectorize(grads)
        grads *= out_to_loss_grad
        grads = grads.sum(dim=0, keepdim=True)
        delta += ch.matmul(grads, XTX_INV).detach()

        olg.append(out_to_loss_grad)

    olg = ch.cat(olg, dim=0)

    delta = delta / len(forget_dataloader)
    delta *= lr
    add_vector_to_parameters(delta.reshape(-1), unlearned_model.parameters())

    return unlearned_model, delta, olg


def trak_dual_projection(
    model: ch.nn.Module,
    xtx_inv_matrix: ch.Tensor,
    proj_dim: int = 8192,
    lr: float = 1.0,
    seed: int = 0,
    train_dataloader: ch.utils.data.DataLoader = None,
    forget_indices: List[int] = None,
    forget_dataloader: ch.utils.data.Subset = None,
) -> ch.nn.Module:
    XTX_INV = xtx_inv_matrix.cuda()
    try:
        model_to_featurize = deepcopy(model)
        unlearned_model = deepcopy(model)
    except NotImplementedError:
        # MiniResNet is annoying to deepcopy
        model_to_featurize = model.custom_deepcopy()
        unlearned_model = model.custom_deepcopy()

    if forget_dataloader is None:
        forget_ds = ch.utils.data.Subset(train_dataloader.dataset, forget_indices)
        forget_dataloader = ch.utils.data.DataLoader(
            forget_ds, batch_size=train_dataloader.batch_size
        )

    grad_dim = sum(p.numel() for p in model_to_featurize.parameters())
    gradient_computer = FunctionalGradientComputer(
        model=model_to_featurize,
        task=ImageClassificationModelOutput(),
        grad_dim=grad_dim,
        dtype=ch.float32,
        device=ch.device("cuda"),
    )
    projector = CudaProjector(
        grad_dim=grad_dim,
        proj_dim=proj_dim,
        seed=seed,
        proj_type=ProjectionType.rademacher,
        device=ch.device("cuda"),
        max_batch_size=32,
    )

    for images, labels in forget_dataloader:
        batch = [images.cuda(), labels.cuda()]
        out_to_loss_grad = gradient_computer.compute_loss_grad(batch)
        grads = gradient_computer.compute_per_sample_grad(batch)
        grads = projector.project(grads, model_id=0)

        grads *= out_to_loss_grad
        grads = grads.sum(dim=0, keepdim=True)
