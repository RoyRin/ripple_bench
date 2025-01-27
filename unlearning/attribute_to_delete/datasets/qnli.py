from argparse import ArgumentParser
import numpy as np
import torch as ch
import random
import torch

from transformers import AutoTokenizer
from datasets import load_dataset


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __getitem__(self, index):
        return self.original_dataset[index] + (index,)

    def __len__(self):
        return len(self.original_dataset)


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def get_qnli_dataset_raw(split, inds=None):
    raw_datasets = load_dataset(
            "glue",
            'qnli',
            cache_dir='/mnt/xfs/datasets/qnli',
            use_auth_token=None,
        )
    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)
    sentence1_key, sentence2_key = task_to_keys['qnli']

    label_to_id = None #{v: i for i, v in enumerate(label_list)}

    tokenizer = AutoTokenizer.from_pretrained(
        'bert-base-cased',
        cache_dir=None,
        use_fast=True,
        revision='main',
        use_auth_token=False
    )

    padding = "max_length"
    max_seq_length=128

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    #with training_args.main_process_first(desc="dataset map pre-processing"):
    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=(not False),
        desc="Running tokenizer on dataset",
    )

    if split == 'train':
        train_dataset = raw_datasets["train"]
        #if data_args.max_train_samples is not None:
        #    max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        #else:
        #    max_train_samples = len(train_dataset)

        #all_indices = range(max_train_samples)
        #train_dataset = train_dataset.select(all_indices)
        ds = train_dataset
    else:
        eval_dataset = raw_datasets["validation"]
        ds = eval_dataset

    return ds


class QNLIDataset:
    def __init__(self, split, inds=None):
        self.dataset = get_qnli_dataset_raw(split, inds)
        self.split = split
        self.size = len(self.dataset)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        idx = int(idx)
        return (np.array(self.dataset[idx]['input_ids']).astype('int32'),
                np.array(self.dataset[idx]['token_type_ids']).astype('int32'),
                np.array(self.dataset[idx]['attention_mask']).astype('int32'),
                self.dataset[idx]['label'])


def get_qnli_dataset(
    split="train", indices=None, raw_imgs=False, indexed=False
):
    is_train = split == "train"
    dataset = QNLIDataset(split)
    if indexed:
        dataset = IndexedDataset(dataset)
    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices)
    return dataset


def get_dataloader(
    dataset, batch_size=256, num_workers=8, indices=None, shuffle=False, indexed=False
):
    if indexed:
        dataset = IndexedDataset(dataset)
    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices)

    return torch.utils.data.DataLoader(
        dataset=dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers
    )


def get_qnli_dataloader(
    batch_size=256,
    num_workers=8,
    split="train",
    shuffle=False,
    indices=None,
    indexed=False,
    drop_last=False
):
    if split == "train_and_val":
        print("Making train-and-val dataset")
        dataset_1 = get_qnli_dataset(split="train", indices=range(10_000), indexed=indexed)
        dataset_2 = get_qnli_dataset(split="val", indexed=indexed)
        dataset = torch.utils.data.ConcatDataset([dataset_1, dataset_2])
        if indices is not None:
            print('TRAIN AND VAL INDICES', indices)
            dataset = torch.utils.data.Subset(dataset, indices)
    else:
        dataset = get_qnli_dataset(
            split=split,indices=indices, indexed=indexed
        )

    loader = torch.utils.data.DataLoader(
        dataset=dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, drop_last=drop_last
    )

    return loader


if __name__ == '__main__':

    loader = get_qnli_dataloader(split='train', shuffle=False)

    all_labels = []
    for batch in loader:
        labels = batch[-1]
        all_labels.append(labels.numpy())

    all_labels = np.concatenate(all_labels)
    np.save('qnli_train_labels.npy', all_labels)
    #import pdb; pdb.set_trace()
    print(batch)
