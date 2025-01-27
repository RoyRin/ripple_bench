import numpy as np
from pathlib import Path

#import torch
from unlearning.auditors.utils import (
    loader_factory,
)
ULIRA_BASE_DIR = Path(
    "/n/home04/rrinberg/data_dir__holylabs/unlearning/precomputed_models/ULIRA_clean/"
)
if not ULIRA_BASE_DIR.exists():
    ULIRA_BASE_DIR = Path("/mnt/xfs/projects/untrak/ULIRA/ULIRA_clean/")



def get_ulira_training_masks(dataset_name):
    masks_path = ULIRA_BASE_DIR / "training_masks.npy"
    masks = np.load(masks_path)
    return masks

ds_name = "CIFAR10"
training_masks = get_ulira_training_masks(ds_name)


def generate_one_ulira_forget_mask(train_targets,
                               training_mask,
                               SEED=42,
                               unlearning_forget_set_count=40,
                               unlearning_forget_set_size=200,
                               class_5_range=1_000):
    """
    class_5_range is the number of points from class 5 we want to consider forgetting
    """
    np.random.seed(SEED)
    class_5_mask = (train_targets == 5)
    N = len(train_targets)

    class_5_indices = class_5_mask.nonzero()[0]
    # pick 1000 points from class 5
    class_5_indices = np.random.choice(class_5_indices,
                                       class_5_range,
                                       replace=False)
    class_5_mask = np.zeros(N)
    class_5_mask[class_5_indices] = 1
    ####
    ulira_forget_mask = []

    indiv_training_mask_ = training_mask


    for _ in range(unlearning_forget_set_count):

        train_and_5 = np.array(indiv_training_mask_ * class_5_mask, dtype=bool)

        class_5_trained_on_mask = np.array(train_and_5)
        class_5_trained_on_indices = class_5_trained_on_mask.nonzero()[0]
        indices = np.random.choice(class_5_trained_on_indices,
                                   unlearning_forget_set_size,
                                   replace=False)
        mask = np.zeros(N)
        mask[indices] = 1
        ulira_forget_mask.append(mask)

    return np.array(ulira_forget_mask)


def get_ulira_forget_masks(dataset_name, original_model_count = 256, class_5_range=1000,unlearnings_per_model = 40, unlearning_forget_set_size=50,overwrite = False ):
    training_mask = get_ulira_training_masks(dataset_name)
    ulira_forget_mask_dir = ULIRA_BASE_DIR / f"forget__{unlearning_forget_set_size}"
    # make dir
    ulira_forget_mask_dir.mkdir(exist_ok=True)

    train_loader = loader_factory(dataset_name, indexed=True)
    train_targets = np.array(train_loader.dataset.original_dataset.targets)


    ulira_forget_masks = []
    for model_i in range(original_model_count):

        path = ulira_forget_mask_dir / f"forget_masks__{model_i}.npy"
        if path.exists() and not overwrite:
            # load it
            #print(f"loading {path}")
            ulira_forget_mask = np.load(path)
            #print(f"ulira_forget_mask.shape {ulira_forget_mask.shape}")
            ulira_forget_masks.append(ulira_forget_mask)

        else:
            training_mask = training_masks[model_i]
            ulira_forget_mask = generate_one_ulira_forget_mask(train_targets, training_mask, class_5_range=class_5_range, unlearning_forget_set_count=unlearnings_per_model)
            # save it
            np.save(path, ulira_forget_mask)
            ulira_forget_masks.append(ulira_forget_mask)
        if model_i % 50 == 0:
            print(f"{model_i} / {original_model_count}")
            #priunt shape
            print(ulira_forget_mask.shape)
    return ulira_forget_masks


def load_ulira_forget_masks(original_model_count=256,
                            unlearning_forget_set_size=50):
    ulira_forget_mask_dir = ULIRA_BASE_DIR / f"forget__{unlearning_forget_set_size}"
    # make dir
    ulira_forget_mask_dir.mkdir(exist_ok=True)

    ulira_forget_masks = []
    for model_i in range(original_model_count):

        path = ulira_forget_mask_dir / f"forget_masks__{model_i}.npy"
        if not path.exists():
            # load it
            #print(f"loading {path}")
            print(f"path.name - {path}")
            raise Exception("path missing")
        ulira_forget_mask = np.load(path)
        #print(f"ulira_forget_mask.shape {ulira_forget_mask.shape}")
        ulira_forget_masks.append(ulira_forget_mask)

        if model_i % 100 == 0:
            print(f"{model_i} / {original_model_count}")
            #priunt shape
            print(ulira_forget_mask.shape)
    return ulira_forget_masks


def load_all_ulira_forget_masks(unlearning_forget_set_size=50):
    ulira_forget_mask_dir = ULIRA_BASE_DIR
    # make dir

    path = ulira_forget_mask_dir / f"all_forget_masks__{unlearning_forget_set_size}.npy"

    if not path.exists():
        # load it
        #print(f"loading {path}")
        raise Exception("path missing")
    return np.load(path)


#ulira_forget_masks = get_ulira_forget_masks(ds_name, class_5_range=1000)
