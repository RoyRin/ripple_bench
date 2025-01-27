import numpy as np
from numpy.lib.format import open_memmap as om
import torch as ch
from typing import List
from copy import deepcopy


class DatamodelWrappedModel(ch.nn.Module):
    def __init__(self, model, trak_scores_on_forget, multiplier):
        super(DatamodelWrappedModel, self).__init__()
        self.model = model
        self.changes_in_prediction = trak_scores_on_forget.sum(dim=0).contiguous()
        device = next(self.model.parameters()).device
        self.changes_in_prediction = self.changes_in_prediction.to(device)
        self.multiplier = multiplier

    def forward(self, x, index=None):
        original_prediciton = self.model(x)
        new_preds = (
            original_prediciton - self.multiplier * self.changes_in_prediction[index]
        )
        return new_preds


def dm_direct(
    model: ch.nn.Module,
    train_dataloader: ch.utils.data.DataLoader = None,
    forget_indices: List[int] = None,
    forget_dataloader: ch.utils.data.DataLoader = None,
    dm_scores_path: str = None,
    reg_path: str = None,
    multiplier: float = 1.0,
    diagonal_only: bool = False,
    estimator: str = "datamodel",
    separate_logit_paths: bool = False,
    **kwargs,
):
    """
    expects trak_scores_matrix to be of shape:
      [train_set_size, train_set_size + val_set_size, num_classes]
    """
    # load trak scores
    if not separate_logit_paths:
        # load trak scores
        trak_scores_matrix = om(dm_scores_path, dtype="float16", mode="r")
        # only load forget set scores in memory
        trak_scores_on_forget = ch.from_numpy(
            np.array(trak_scores_matrix[forget_indices])
        )
    else:
        # load trak scores
        # hacky for cifar-10
        trak_scores_matrices = {}
        for i in range(10):
            trak_scores_matrices[i] = om(f"{dm_scores_path}/logit_{i}.mmap", mode="r")
            # only load forget set scores in memory
        trak_scores_on_forget = ch.stack(
            [
                ch.from_numpy(np.array(trak_scores_matrices[i][forget_indices]))
                for i in range(10)
            ],
            dim=-1,
        )
        print("trak_scores_on_forget.shape", trak_scores_on_forget.shape)

    if diagonal_only:
        trak_scores_on_forget_zeroed = trak_scores_on_forget.clone()
        trak_scores_on_forget_zeroed[np.arange(len(forget_indices)), forget_indices] = (
            0.0
        )
        trak_scores_on_forget -= trak_scores_on_forget_zeroed

    # unwrap model
    _base_model = deepcopy(model.model)
    _base_model.eval()

    # (NOT USED ATM) "un-softhreshold" using reg param
    if reg_path is not None:
        lam = np.load(reg_path)
        NUM_CLASSES = trak_scores_on_forget.shape[-1]
        trak_scores_on_forget -= (trak_scores_on_forget < 0).float() * lam
        trak_scores_on_forget += (trak_scores_on_forget > 0).float() * lam

    # wrap it again, subtracting the changes in prediction
    unlearned_model = DatamodelWrappedModel(
        _base_model, trak_scores_on_forget, multiplier
    )
    return unlearned_model
