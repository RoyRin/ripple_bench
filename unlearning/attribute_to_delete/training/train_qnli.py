from argparse import ArgumentParser
from typing import List
import time
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os

import torch
import torch as ch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Conv2d, BatchNorm2d
from torch.optim import SGD, AdamW, lr_scheduler
import torchvision

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from unlearning.datasets.qnli import get_qnli_dataloader


NUM_TRAIN = 10_000


'''
Section('training', 'Hyperparameters').params(
    lr=Param(float, 'The learning rate to use', default=2e-3),
    epochs=Param(int, 'Number of epochs to run for', default=8),
    lr_peak_epoch=Param(int, 'Peak epoch for cyclic lr', default=5),
    batch_size=Param(int, 'Batch size', default=512),
    momentum=Param(float, 'Momentum for SGD', default=0.9),
    weight_decay=Param(float, 'l2 weight decay', default=5e-4),
    label_smoothing=Param(float, 'Value of label smoothing', default=0.0),
    num_workers=Param(int, 'The number of workers', default=12),
    model_num=Param(int, 'Model number', required=True),
    optimizer=Param(str, 'optimizer', default='sgd'),
)

Section('cfg', 'config').params(
    out_dir=Param(str, default=None)
)

Section('data', 'data related stuff').params(
    alpha=Param(float, 'subsampling', default=1.0)
)
'''


def get_logits(model, loader: torch.utils.data.DataLoader):
    training = model.training
    model.eval()
    all_logits = torch.zeros(len(loader.dataset), 2)
    batch_size = loader.batch_size
    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader)):
            batch = [x.cuda() for x in batch]
            input_ids, token_type_ids, attention_mask, labels = batch
            with autocast():
                logits = model(input_ids=input_ids, token_type_ids=token_type_ids,
                        attention_mask=attention_mask).logits
            all_logits[i * batch_size : (i + 1) * batch_size] = logits.cpu()

    model.train(training)
    return all_logits


def compute_margin(logit, true_label):
    logit_other = logit.clone()
    logit_other[true_label] = -np.inf

    return logit[true_label] - logit_other.logsumexp(dim=-1)


def get_margins(targets,  logits):
    margins = [
        compute_margin(logit, target)
        for (logit, target) in zip(logits, targets)
    ]
    return np.array(margins)


def wrapper_for_train_qnli_on_subset_submitit(
        masks_path: str,
        idx_start: int,
        n_models: int,
        ckpt_dir: str,
        should_save_train_logits: bool = False,
        should_save_val_logits: bool = False,
        model_id_offset: int = 0,
        evaluate = True,
        checkpoint_epochs=[]):
    """
    - masks_path gives the path to a np array of shape [K, train_set_size],
    where each row gives us a boolean mask of the samples that we want to train
    on (True for the included samples and False for the excluded samples)

    - idx_start and n_models tell us which rows of the masks to use for training;
    in particular, we will use the masks from masks_path[idx_start:idx_start +
    n_models] for training

    - model_id_offset is the offset to add to the model_id when saving the
    model, here only for a hacky use, feel free to ignore
    """

    if masks_path == "":
        print("No masks path given, using all samples")
        all_masks = np.ones((n_models, 10000), dtype=bool)
    else:
        all_masks = np.load(masks_path)

    print(f"computing from {idx_start} to {idx_start + n_models - 1}")
    for model_id in range(idx_start, idx_start + n_models):
        print(f"training - {model_id}")
        print(f"ckpt_dir-{ckpt_dir}")
        if (ckpt_dir / f"val_logits_{model_id + model_id_offset}.pt").exists():
            print(f"skipping model {model_id} because logits already exist")
            continue
        mask = all_masks[model_id]
        print(f"mask- {mask}")
        indices = np.where(mask)[0]
        assert len(indices) > 0, "mask is empty"
        # HACK by ROY, to confirm that we are training 50%-ers
        #assert len(indices) != 50000, "training full models."

        loader = get_qnli_dataloader(indices=indices,
                                      shuffle=True,
                                      num_workers=2,
                                      indexed=False)
        eval_loader = get_qnli_dataloader(split="val",
                            shuffle=True,
                            num_workers=2,
                            indexed=False)

        model = construct_model()

        train(model=model,
                train_loader=loader,
                checkpoints_dir=ckpt_dir,
                model_id=model_id + model_id_offset,
                checkpoint_epochs=checkpoint_epochs)

        # eval model
        if evaluate:
            evaluate_model(model, eval_loader)
            #print(f"eval acc-  {acc}")
            #print("------")

        if should_save_train_logits:
            full_dataloader_unshuffled = get_qnli_dataloader(num_workers=2,
                                                             indices=range(NUM_TRAIN),
                                                              indexed=False)
            logits = get_logits(model, full_dataloader_unshuffled)
            logits_path = ckpt_dir / f"train_logits_{model_id + model_id_offset}.pt"
            torch.save(logits, logits_path)
            print(f"saved logits to {logits_path}")
        if should_save_val_logits:
            val_dataloader_unshuffled = get_qnli_dataloader(split="val",
                                                             num_workers=2,
                                                             indexed=False)
            val_logits = get_logits(model, val_dataloader_unshuffled)
            val_logits_path = ckpt_dir / f"val_logits_{model_id + model_id_offset}.pt"
            torch.save(val_logits, val_logits_path)
            print(f"saved logits to {val_logits_path}")
    return 0  # return 0 if everything went well


def construct_model():
    config = AutoConfig.from_pretrained(
        'bert-base-cased',
        num_labels=2,
        finetuning_task='qnli',
        cache_dir=None,
        revision='main',
        use_auth_token=None,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-cased',
        config=config,
        cache_dir=None,
        revision='main',
        use_auth_token=None,
        ignore_mismatched_sizes=False,
        #output_hidden_states=True
    )

    model = model.cuda()
    model = model.train()
    model.bert.pooler.activation = ch.nn.Identity()

    #import ipdb; ipdb.set_trace()
    return model


def train(model,
          train_loader,
          lr=2e-3,
          epochs=10,
          label_smoothing=0.,
          momentum=0.9,
          weight_decay=5e-4,
          lr_peak_epoch=None,
          model_num=None,
          optimizer='sgd',
          out_dir=None,
          checkpoint_epochs=[],
          checkpoints_dir=Path("./data/qnli_checkpoints"),
          overwrite=False,
          model_id=0,
          model_save_suffix=""
    ):
    if optimizer == 'sgd':
        opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        opt = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.)

    iters_per_epoch = len(train_loader)
    # Cyclic LR with single triangle
    #lr_schedule = np.ones((epochs+1) * iters_per_epoch)

    lr_schedule = np.interp(np.arange((epochs+1) * iters_per_epoch),
                            [0, epochs * iters_per_epoch],
                            [1, 0])

    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)

    for ep in range(epochs):
        #os.makedirs(Path(out_dir) / f'model_{model_num}', exist_ok=True)
        #ch.save(model.state_dict(), Path(out_dir) / f'model_{model_num}/model_sd_{ep}.pt')
        model_count = 0
        for it, batch in enumerate(tqdm(train_loader)):
            # Save a checkpoint every 100 iterations
            # if (ep * iters_per_epoch + it) % 100 == 0:
                # ch.save(model.state_dict(), f'../results/model_4/model_sd_{model_count}.pt')
                # model_count += 1

            opt.zero_grad(set_to_none=True)
            batch = [x.cuda() for x in batch]
            input_ids, token_type_ids, attention_mask, labels = batch

            with autocast():
                #print(input_ids.dtype, token_type_ids.dtype, attention_mask.dtype, labels.dtype)
                logits = model(input_ids=input_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask).logits
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

        if ep in checkpoint_epochs:
            print(f"saving model at epoch {ep}")
            ch.save(
                model.state_dict(),
                checkpoints_dir / f"sd_{model_id}__{model_save_suffix}__epoch_{ep}.pt",
            )
            #if should_save_logits:
            #    print("saving logits !!!!!!!")
            #    save_logits_and_margins(model, checkpoints_dir, model_id, ep)


def evaluate_model(model, loader, ret_vals=False):
    model.eval()
    with ch.no_grad():
        total_correct, total_num = 0., 0.
        all_logits = []
        for batch in tqdm(loader):
            batch = [x.cuda() for x in batch]
            input_ids, token_type_ids, attention_mask, labels = batch

            with autocast():
                logits = model(input_ids=input_ids, token_type_ids=token_type_ids,
                        attention_mask=attention_mask).logits
                #print(logits)
                all_logits.append(logits.cpu())
                total_correct += logits.argmax(1).eq(labels).sum().cpu().item()
                total_num += input_ids.shape[0]
        all_logits = ch.cat(all_logits, 0).numpy()
        print(f'Accuracy: {total_correct / total_num * 100:.1f}%')
    if ret_vals:
        return all_logits


if __name__ == '__main__':

    loaders = {
        'train': get_qnli_dataloader(indices=range(NUM_TRAIN // 2),
                                    shuffle=True,
                                    num_workers=2,
                                    indexed=False),
        'val': get_qnli_dataloader(split='val',
                                    shuffle=False,
                                    num_workers=2,
                                    indexed=False),
        }

    model = construct_model()
    train(model, loaders['train'], epochs=10)
    evaluate_model(model, loaders['train'])
    evaluate_model(model, loaders['val'])

    logits = get_logits(model, loaders['val'])
    logits_path = f"sample_val_logits.pt"
    torch.save(logits, logits_path)