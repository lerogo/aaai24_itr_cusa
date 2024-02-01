import os
from typing import Callable

import torch
from torch.utils.data import DataLoader

from dataset.cross_coco_dataset import cross_coco_dataset, cross_coco_test_dataset


def get_dataset(datasetName: str, dataPath: str,  split: str, transform: Callable):
    if datasetName == "cross_coco" or datasetName == "cross_flickr":
        if split == "train" or split == "experiment":
            return cross_coco_dataset(dataPath, transform, split)
        elif split == "test" or split == "val":
            return cross_coco_test_dataset(dataPath, transform, split)
    else:
        raise NotImplementedError(
            "Dataset {} not implemented.".format(datasetName))


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders
