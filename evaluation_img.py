import argparse
import os
from typing import Dict, Tuple

import numpy as np
import torch
from torch.nn.functional import normalize
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset

import tqdm


def get_dataset(dataset_name: str, transformer=None) -> Dict:
    # this is the root of the dataset
    # Note: 
    #   We have extracted the region feature for each image 
    #   and saved it in the same location as the image, 
    #   named "<ImageFileName>.npz".
    #   The region feature is for sgraf model in the paper
    root = ""

    if dataset_name == "sop":
        from dataset_evalimg import SOP
        trainset = SOP(root, "train",transform=transformer)
        testset = SOP(root, "eval",transform=transformer)
        trainset.num_classes = trainset.nb_classes()
        return {"train": trainset, "test": testset, "metric": "rank1"}

    elif dataset_name == "cub":
        from dataset_evalimg import CUBirds
        trainset = CUBirds(root, "train",transform=transformer)
        testset = CUBirds(root, "eval",transform=transformer)
        trainset.num_classes = trainset.nb_classes()
        return {"train": trainset, "test": testset, "metric": "rank1"}

    elif dataset_name == "car":
        from dataset_evalimg import Cars
        trainset = Cars(root, "train",transform=transformer)
        testset = Cars(root, "eval",transform=transformer)
        trainset.num_classes = trainset.nb_classes()
        return {"train": trainset, "test": testset, "metric": "rank1"}

    elif dataset_name == "inshop":
        from dataset_evalimg import Inshop_Dataset
        trainset = Inshop_Dataset(root, "train",transform=transformer)
        query = Inshop_Dataset(root, "query",transform=transformer)
        gallery = Inshop_Dataset(root, "gallery",transform=transformer)
        trainset.num_classes = trainset.nb_classes()
        return {"train": trainset, "query": query, "gallery": gallery, "metric": "rank1"}
    
    elif dataset_name == "inat":
        from dataset_evalimg import inaturalist
        trainset = inaturalist.get_trainset(root, transform=transformer)
        testset = inaturalist.get_testset(root, transform=transformer)
        trainset.num_classes = 5690
        return {"train": trainset, "test": testset, "metric": "rank1"}
    else:
        raise


@torch.no_grad()
def extract_feat(
        model: torch.nn.Module,
        dataset: Dataset,
        batch_size: int,
        num_workers: int) -> Tuple[torch.Tensor, torch.Tensor]:
    n_data = len(dataset)
    idx_all_rank = list(range(n_data))
    dataset_this_rank = Subset(dataset, idx_all_rank)
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "drop_last": False
    }
    dataloader = DataLoader(dataset_this_rank, **kwargs)
    x = None
    y_np = []
    idx = 0
    for image, label in tqdm.tqdm(dataloader):
        image = image.cuda()
        embedding = get_feature(model, image)
        embedding_size: int = embedding.size(1)
        if x is None:
            size = [len(idx_all_rank), embedding_size]
            x = torch.zeros(*size, device=image.device)
        x[idx:idx + embedding.size(0)] = embedding
        y_np.append(np.array(label))
        idx += embedding.size(0)
    x = x.cpu()
    y_np = np.concatenate(y_np, axis=0)

    return x, y_np


@torch.no_grad()
def euclidean_distance(x, y, topk=2):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(mat1=x, mat2=y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()
    return torch.topk(dist, topk, largest=False)


@torch.no_grad()
def get_metric(
        query: torch.Tensor,
        query_label: list,
        gallery: torch.Tensor = None,
        gallery_label: list = None,
        l2norm=True,
        metric="rank1"):

    if gallery is None:
        query = query.cuda()
        if l2norm:
            query = normalize(query)
        query_label = query_label
        list_pred = []
        num_feat = query.size(0)

        idx = 0
        is_end = 0
        while not is_end:
            if idx + 128 < num_feat:
                end = idx + 128
            else:
                end = num_feat
                is_end = 1

            _, index_pt = euclidean_distance(query[idx:end], query)
            index_np = index_pt.cpu().numpy()[:, 1]
            list_pred.append(index_np)
            idx += 128
        query_label = np.array(query_label).reshape(num_feat)
        pred = np.concatenate(list_pred).reshape(num_feat)
        rank_1 = np.sum(query_label == query_label[pred]) / num_feat
        rank_1 = float(rank_1)
        return rank_1 * 100
    else:
        query = query.cuda()
        query_label = query_label
        gallery = gallery.cuda()
        gallery_label = np.array(gallery_label)
        list_pred = []
        if l2norm:
            query = normalize(query)
            gallery = normalize(gallery)
        num_feat = query.size(0)

        idx = 0
        is_end = 0
        while not is_end:
            if idx + 128 < num_feat:
                end = idx + 128
            else:
                end = num_feat
                is_end = 1

            _, index_pt = euclidean_distance(query[idx:end], gallery)
            index_np = index_pt.cpu().numpy()[:, 0]

            list_pred.append(index_np)
            idx += 128
        query_label = np.array(query_label).reshape(num_feat)
        pred = np.concatenate(list_pred).reshape(num_feat)
        rank_1 = np.sum(query_label == gallery_label[pred]) / num_feat
        rank_1 = float(rank_1)
        return rank_1 * 100


@torch.no_grad()
def evaluation(model: torch.nn.Module,
               dataset_dict: Dict, batch_size: int, num_workers: int):

    if "index" in dataset_dict:
        raise NotImplementedError
    elif "test" in dataset_dict:
        dataset = dataset_dict["test"]
        x, y = extract_feat(model, dataset, batch_size, num_workers)
        metric = get_metric(x, y)
        return metric

    elif "query" in dataset_dict and "gallery" in dataset_dict:
        dataset_q = dataset_dict["query"]
        dataset_g = dataset_dict["gallery"]
        q, q_label = extract_feat(model, dataset_q, batch_size, num_workers)
        g, g_label = extract_feat(model, dataset_g, batch_size, num_workers)
        metric = get_metric(query=q, query_label=q_label,
                            gallery=g, gallery_label=g_label)
        return metric


def get_feature(model, image):
    # return model.encode_image(image)
    # return model.encode_image(image, cross_modal=True)
    return model.encode_image(image, cross_modal=False)


def get_model(device, modelName):
    from unire.model import unire
    checkPointPath = modelName
    print("Loading model: ", checkPointPath)
    checkpoint = torch.load(checkPointPath, map_location='cpu')
    state_dict = checkpoint['model']
    args = argparse.Namespace()
    args.gpu = torch.device(device)
    model = unire(args, checkpoint['config'])
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model, model.preprocess

# ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']

def get_clip_model(device):
    modelName = "ViT-L/14@336px"
    print("Loading model: ", modelName)
    from clip import clip
    clip_model, preprocess = clip.load(modelName, device=device, jit=False)
    clip_model.eval()
    return clip_model, preprocess


if __name__ == '__main__':
    processList = []
    fileList = list(os.walk("output/vitb32/coco"))
    for val in fileList:
        if "checkpoint_best.pth" in val[2]:
            processList.append(os.path.join(val[0], "checkpoint_best.pth"))

    for modelName in processList:
        datasetNameList = ["cub","sop", "inshop","inat"]
        batch_size = 128
        num_workers = 4

        model, preprocess = get_model("cuda:0", modelName)

        scores = []

        for datasetName in datasetNameList:
            print(datasetName)
            dataset_dict: Dict = get_dataset(datasetName, preprocess)
            score = evaluation(model, dataset_dict, batch_size, num_workers)
            scores.append(score)
            if isinstance(score, Tuple):
                for i in score:
                    print(i, end=",")
            else:
                print(score, end=",")
            print("\n")
        print("scores: ", scores)
        print("mean score: ", np.mean(scores))
