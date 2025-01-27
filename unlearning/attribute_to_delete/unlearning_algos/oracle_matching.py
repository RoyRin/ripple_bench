import os
import numpy as np
import torch
import torch as ch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import autocast

from typing import List
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
from unlearning.auditors.utils import BASE_DIR
import copy
from numpy.lib.format import open_memmap as om

from .dm_direct import DatamodelWrappedModel
from unlearning.unlearning_algos.utils import construct_loader
from unlearning.datasets import DATASET_SIZES
from unlearning.auditors.utils import load_model_from_dict
#import clip

def to_cuda(x):
    if isinstance(x, (list, tuple)):
        return [to_cuda(xi) for xi in x]
    return x.cuda()


def finetune(
    model,
    oracles,
    dataloader,
    train_and_val_loader,
    optimizer,
    num_epochs,
    learning_rate,
    wd_lambda=0.0,
    loss_type='MSE',
    save_all_outputs=False,
    save_iter=30,
):
    print('NUM ORACLES:', len(oracles))
    # Weight decay towards original params
    orig_params = [copy.deepcopy(p) for p in model.parameters()]
    def wd_reg(model, reg=wd_lambda):
        loss = 0
        for orig_p, p in zip(orig_params, model.parameters()):
            loss += ch.norm(p - orig_p) ** 2.0
        return loss * reg

    if optimizer == "adam":
        opt = ch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "adamw":
        opt = ch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        opt = ch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)

    if save_all_outputs:
        # For analyzing entire trajectories.
        all_oracle_outs = []
        all_model_outs = []
        for x, y, _ in train_and_val_loader:
            with ch.no_grad():
                x = x.cuda()
                oracle_outs = ch.stack([oracle(x) for oracle in oracles]).mean(dim=0)
            all_oracle_outs.append(oracle_outs.cpu())
        all_oracle_outs = ch.cat(all_oracle_outs)

        # Before training
        model.eval()
        model_outs = []
        for x, y, _ in train_and_val_loader:
            with ch.no_grad():
                x = x.cuda()
                outs = model(x)
            model_outs.append(outs.cpu())
        model_outs = ch.cat(model_outs)
        all_model_outs.append(model_outs)

    for epoch in range(num_epochs):
        model.train()

        pbar = tqdm(enumerate(dataloader))
        avg_loss = 0.
        for it, (x, _, index) in pbar:
            opt.zero_grad()
            x = to_cuda(x)

            with autocast():
                logits = model(x, index)
                with ch.no_grad():

                    if loss_type == 'MSE':
                        oracle_outs = ch.stack([oracle(x, index) for oracle in oracles]).mean(
                            dim=0
                        )
                    elif loss_type == 'CE':
                        oracle_probs = ch.stack([ch.nn.functional.softmax(oracle(x, index), dim=-1) for oracle in oracles]).mean(
                            dim=0
                        )
                    else:
                        raise ValueError(f"Unknown loss_type: {loss_type}")
            if loss_type == 'MSE':
                loss = ch.nn.functional.mse_loss(oracle_outs, logits) + wd_reg(model)
            elif loss_type == 'CE':
                loss = ch.nn.functional.cross_entropy(logits, oracle_probs) + wd_reg(model)
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            loss.backward()
            opt.step()

            pbar.set_description(f"total loss: {loss.item():.5f}")
            avg_loss += loss.item()

            if save_all_outputs and (it+1) % save_iter == 0:
                model.eval()
                model_outs = []
                for x, y, _ in train_and_val_loader:
                    with ch.no_grad():
                        x = to_cuda(x)
                        outs = model(x)
                    model_outs.append(outs.cpu())
                model_outs = ch.cat(model_outs)
                all_model_outs.append(model_outs)
                model.train()

        scheduler.step()
        print("Epoch:", epoch, "| avg_loss:", avg_loss / len(dataloader), flush=True)

    print('Final weight norm:', wd_reg(model, reg=1.0))

    if save_all_outputs:
        all_model_outs = ch.stack(all_model_outs)
        return all_oracle_outs, all_model_outs
    return None


def duplicate_forget_indices(forget_indices, forget_multiplier):
    '''
        Duplicate forget_indices by [forget_multiplier]
    '''
    forget_indices = forget_indices.tolist()
    forget_multiplier_int = int(forget_multiplier)
    forget_multiplier_rem = forget_multiplier - forget_multiplier_int
    num_take = int(len(forget_indices) * forget_multiplier_rem)

    forget_indices_duplicated = forget_indices * forget_multiplier_int
    forget_indices_duplicated.extend(forget_indices[:num_take])

    return forget_indices_duplicated


def train_knowledge_distillation(model,
                                 oracles,
                                 dataloader,
                                 optimizer,
                                 num_epochs,
                                 learning_rate,
                                 T=2,
                                 soft_target_loss_weight=.25,
                                 ce_loss_weight=.75,
                                 **kwargs):
    ce_loss = nn.CrossEntropyLoss()

    #teacher.eval()  # Teacher set to evaluation mode
    print(f"train_knowledge_distillation!")
    model.train()  # Student to train mode

    for epoch in range(num_epochs):
        lr = learning_rate * 0.95 ** (epoch + 1)
        if optimizer == "adam":
            opt = ch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer == "adamw":
            opt = ch.optim.AdamW(model.parameters(), lr=lr)
        elif optimizer == "sgd":
            opt = ch.optim.SGD(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        running_loss = 0.0
        #for inputs, labels in train_loader:

        pbar = tqdm(dataloader)
        for x, labels, index in pbar:
            #inputs, labels = inputs.to(device), labels.to(device)
            x = x.cuda()
            # this is a bit contreversial - do we want to learn on the ground truth or not? for the forget set...
            #labels = labels.cuda()

            student_logits = model(x, index)

            opt.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights

            with ch.no_grad():
                teacher_logits = ch.stack(
                    [oracle(x, index) for oracle in oracles]).mean(dim=0)
            #with ch.no_grad():
            #    teacher_logits = teacher(inputs)

            # Forward pass with the student model
            #student_logits = student(inputs)

            #Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

            oracle_labels = ch.argmax(teacher_logits, dim=1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = ch.sum(
                soft_targets *
                (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**
                                                                           2)

            # Calculate the true label loss
            label_loss = ce_loss(
                student_logits, oracle_labels
            )  # compare loss to oracle labels and oracle logits

            # Weighted sum of the two losses
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()
            opt.step()

            running_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(dataloader)}"
        )


def oracle_matching(
    model: ch.nn.Module,
    train_dataloader: ch.utils.data.DataLoader = None,
    forget_indices: List[int] = None,
    forget_dataloader: ch.utils.data.DataLoader = None,
    oracles_path: str = None,
    loss_type: str = 'MSE',
    num_epochs: int = 5,
    learning_rate: float = 5e-4,
    wd_lambda: float = 0.0,
    #retain_learning_rate: float = 5e-4,
    batch_size: int = 512,
    optimizer: str = "adam",
    retain_multiplier: float = 3.0,
    forget_multiplier: float = 1.0,
    num_oracles: int = 10,
    shuffle: bool = False,
    seed: int = 42,
    save_all_outputs: bool = False,
    train_and_val_loader: ch.utils.data.DataLoader = None,
    dataset='CIFAR10',
    **kwargs,
):
    """
    avg_logits_path is a path to a tensor
    of shape [train_set_size]

    retain_data_amount defines how much of the retain set
    to use for finetuning as a multiple of the forget set size
    """
    unlearned_model = deepcopy(model)
    unlearned_model.train()

    num_retain_samples = int(retain_multiplier * len(forget_indices))
    N = DATASET_SIZES[dataset.upper()]
    retain_set_indices = np.setdiff1d(
        np.arange(N), forget_indices
    )
    rng = np.random.default_rng(seed=seed)
    randomly_selected_indices = rng.choice(retain_set_indices, num_retain_samples,
                                           replace=False)
    print(randomly_selected_indices)

    forget_indices_duplicated = duplicate_forget_indices(forget_indices, forget_multiplier)
    indices_to_finetune_on = np.concatenate(
         [np.array(forget_indices_duplicated), randomly_selected_indices]
    )

    finetuning_dataloader = construct_loader(dataset, train_dataloader, indices_to_finetune_on, batch_size, 8, shuffle)

    oracles = []
    paths = list(Path(oracles_path).rglob("sd_*.pt"))

    np.random.shuffle(paths)
    for oracle_path in paths:
        oracle = deepcopy(model)
        load_model_from_dict(oracle, oracle_path)
        oracle = oracle.cuda().eval()
        oracles.append(oracle)
        if len(oracles) == num_oracles:
            break

    # finetune on a mix of forget set and retain set
    result = finetune(
        unlearned_model,
        oracles,
        finetuning_dataloader,
        train_and_val_loader,
        optimizer,
        num_epochs,
        learning_rate,
        wd_lambda,
        loss_type,
        save_all_outputs=save_all_outputs,
    )

    if save_all_outputs:
        oracle_outs, model_outs = result
        return unlearned_model, {'oracle_logits': oracle_outs, 'model_logits': model_outs}
    return unlearned_model


def dm_matching(
    model: ch.nn.Module,
    train_dataloader: ch.utils.data.DataLoader = None,
    forget_indices: List[int] = None,
    forget_dataloader: ch.utils.data.DataLoader = None,
    oracles_path: str = None,
    loss_type: str = 'MSE',
    num_epochs: int = 5,
    learning_rate: float = 5e-4,
    wd_lambda: float = 0.0,
    #retain_learning_rate: float = 5e-4,
    batch_size: int = 512,
    optimizer: str = "adam",
    dm_multiplier: float = 1.0,
    retain_multiplier: float = 3.0,
    forget_multiplier: float = 1.0,
    shuffle: bool = False,
    seed: int = 42,
    save_all_outputs: bool = False,
    train_and_val_loader: ch.utils.data.DataLoader = None,
    dataset='CIFAR10',
    **kwargs,
):
    """
    avg_logits_path is a path to a tensor
    of shape [train_set_size]

    retain_data_amount defines how much of the retain set
    to use for finetuning as a multiple of the forget set size
    """
    dm_matrix = om(oracles_path, dtype="float16", mode="r")
    # only load forget set scores in memory
    dm_on_forget = ch.from_numpy(np.array(dm_matrix[forget_indices]))
    dm_oracle_model = DatamodelWrappedModel(model, dm_on_forget, dm_multiplier)

    unlearned_model = deepcopy(model)
    unlearned_model.train()

    num_retain_samples = int(retain_multiplier * len(forget_indices))
    retain_set_indices = np.setdiff1d(
        np.arange(len(train_dataloader.dataset)), forget_indices
    )
    rng = np.random.default_rng(seed=seed)
    randomly_selected_indices = rng.choice(retain_set_indices, num_retain_samples,
                                           replace=False)
    print(randomly_selected_indices)

    forget_indices_duplicated = duplicate_forget_indices(forget_indices, forget_multiplier)
    indices_to_finetune_on = np.concatenate(
         [np.array(forget_indices_duplicated), randomly_selected_indices]
    )

    finetuning_dataloader = construct_loader(dataset, train_dataloader, indices_to_finetune_on, batch_size, 8, shuffle)

    oracles = [dm_oracle_model]

    # finetune on a mix of forget set and retain set
    result = finetune(
        unlearned_model,
        oracles,
        finetuning_dataloader,
        train_and_val_loader,
        optimizer,
        num_epochs,
        learning_rate,
        wd_lambda,
        loss_type,
        save_all_outputs=save_all_outputs,
    )

    if save_all_outputs:
        oracle_outs, model_outs = result
        return unlearned_model, {'oracle_logits': oracle_outs, 'model_logits': model_outs}
    return unlearned_model


def get_embedding(embeddings_save_path = 'cifar10_clip_embeddings.pt'):
    if not os.path.exists(embeddings_save_path):

        # Load the model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, preprocess = clip.load('ViT-B/32', device=device)

        # Load the CIFAR-10 dataset
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            preprocess
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False, num_workers=2)

        # Function to compute embeddings
        def compute_clip_embeddings(dataloader):
            embeddings = []
            with torch.no_grad():
                for images, _ in tqdm(dataloader):
                    images = images.to(device)
                    image_features = clip_model.encode_image(images)
                    embeddings.append(image_features)

            return torch.cat(embeddings)

        # Compute embeddings
        embeddings = compute_clip_embeddings(trainloader)
        embeddings = embeddings.cpu().numpy()

        # Save embeddings for later use
        torch.save(embeddings, embeddings_save_path)
        print("Embeddings computed and saved.")
    else:
        print(f"loading embeddings")
        embeddings = torch.load(embeddings_save_path)
        print("Embeddings loaded.")
    return embeddings


def finetune_till_zero_loss(model,
                            oracles,
                            dataloader,
                            optimizer,
                            learning_rate,
                            delta,
                            min_epochs,
                            num_epochs=None,
                            **kwargs):
    if num_epochs is None:
        num_epochs = min_epochs * 5
    prev_loss = float('inf')
    for epoch in range(num_epochs):
        lr = learning_rate * 0.95**(epoch + 1)
        if optimizer == "adam":
            opt = ch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer == "adamw":
            opt = ch.optim.AdamW(model.parameters(), lr=lr)
        elif optimizer == "sgd":
            opt = ch.optim.SGD(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        pbar = tqdm(dataloader)
        for x, _, index in pbar:
            opt.zero_grad()
            x = x.cuda()
            logits = model(x)
            with ch.no_grad():
                oracle_outs = ch.stack([oracle(x)
                                        for oracle in oracles]).mean(dim=0)
            loss = ch.nn.functional.mse_loss(oracle_outs, logits)
            loss.backward()
            opt.step()
            pbar.set_description(f"loss: {loss.item():.2f}")

        if epoch >= min_epochs and abs(loss - prev_loss) < delta:
            break

        prev_loss = loss

