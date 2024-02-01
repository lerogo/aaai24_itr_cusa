"""
ECCV Caption
Copyright (c) 2022-present NAVER Corp.
MIT license
"""
import argparse
import os
import re
import clip
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO

from eccv_caption import Metrics


def pre_caption(caption, max_words=64):
    caption_raw = caption
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        ' ',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    if not len(caption):
        raise ValueError(f"pre_caption yields invalid text (raw: {caption_raw})")

    return caption


def run(
        model_path,
        coco_img_root='', 
        coco_ann_path='',):
    """
    Run the evaluation on the ECCV Caption dataset
    :param model_path: path to the model checkpoint
    :param coco_img_root: path to the COCO images, all images should be in the same folder
    :param coco_ann_path: path to the COCO annotations
    :return: None
    """
    # Prepare metric
    metric = Metrics()

    # Prepare the inputs
    coco = COCO(coco_ann_path)
    test_cids = metric.coco_ids

    # Load the model
    device = "cuda"
    model, preprocess = get_model(device, model_path)

    all_image_features = []
    all_text_features = []
    all_iids, all_cids = [], []
    seen_iids = set()
    with torch.no_grad():
        for cid in tqdm(test_cids):
            iid = int(coco.anns[cid]['image_id'])
            if iid not in seen_iids:
                path = coco.imgs[iid]['file_name']
                image = Image.open(os.path.join(coco_img_root, path)).convert('RGB')
                image_input = preprocess(image).unsqueeze(0).to(device)

                # Calculate features
                image_features = image2feature(model, image_input, device)
                all_image_features.append(image_features[0])
                all_iids.append(iid)
                seen_iids.add(iid)

            # text_inputs = clip.tokenize(coco.anns[cid]['caption']).to(device)
            # text_inputs = sentence2feature(model, coco.anns[cid]['caption'], device, max_length=77)

            # Calculate features
            caption = pre_caption(coco.anns[cid]['caption'])
            text_features = sentence2feature(model, caption, device, max_length=77)
            all_text_features.append(text_features[0])
            all_cids.append(int(cid))

    image_features = torch.stack(all_image_features, dim=0)
    text_features = torch.stack(all_text_features, dim=0)
    print(image_features.shape, text_features.shape)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    sims = image_features @ text_features.T
    print("sims.shape", sims.shape)

    i2t = {}
    t2i = {}

    all_cids = np.array(all_cids)
    all_iids = np.array(all_iids)

    # 50 is enough for ECCV metrics (max ECCV t2i/i2t positives = 19/48)
    # In the main paper, we use the modified PMRP by using K = 50.
    # If you want to use the original PMRP, then K should be larger than 13380
    # (max PM t2i/i2t positives = 2676/13380)

    K = 50
    for idx, iid in enumerate(all_iids):
        values, indices = sims[idx, :].topk(K)
        indices = indices.detach().cpu().numpy()
        i2t[iid] = [int(cid) for cid in all_cids[indices]]

    for idx, cid in enumerate(all_cids):
        values, indices = sims[:, idx].topk(K)
        indices = indices.detach().cpu().numpy()
        t2i[cid] = [int(iid) for iid in all_iids[indices]]

    # print(i2t, t2i)

    scores = metric.compute_all_metrics(
        i2t, t2i,
        target_metrics=('eccv_r1', 'eccv_map_at_r', 'eccv_rprecision', 'coco_5k_recalls', 'cxc_recalls'),
        Ks=(1, 5, 10),
        verbose=False
    )

    for key in scores:
        scores[key]['i2t'] = scores[key]['i2t'] * 100
        scores[key]['t2i'] = scores[key]['t2i'] * 100
    import json
    print(json.dumps(scores))

def get_clip_model(device, model_path):
    # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    from clip import clip
    modelName = model_path
    print("Loading model: ", modelName)
    clip_model, preprocess = clip.load(modelName, device=device, jit=False)
    clip_model.eval()
    return clip_model, preprocess


def sentence2feature_clip(model, sentences, device, max_length):
    with torch.no_grad():
        from clip import clip
        if max_length is None:
            tokens = clip.tokenize(sentences, truncate=True)
        else:
            tokens = clip.tokenize(sentences, context_length=max_length, truncate=True)
        tokens = tokens.to(device)
        return model.encode_text(tokens)


def image2feature_clip(model, images, device):
    return model.encode_image(images)


def get_model(device, model_path):
    # return get_clip_model(device, model_path)
    from unire.model import unire
    checkPointPath = model_path
    print("Loading model: ", checkPointPath)
    checkpoint = torch.load(checkPointPath, map_location='cpu')
    state_dict = checkpoint['model']
    args = argparse.Namespace()
    args.gpu = torch.device(device)
    model = unire(args, checkpoint['config'])
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    # Statistical Model Parameter Million
    print("Model Parameter Million: ", sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values()) / 1000000.0)
    return model, model.preprocess


def sentence2feature(model, sentences, device, max_length):
    # return sentence2feature_clip(model, sentences, device, max_length)
    with torch.no_grad():
        from clip import clip
        if max_length is None:
            tokens = clip.tokenize(sentences, truncate=True)
        else:
            tokens = clip.tokenize(sentences, context_length=max_length, truncate=True)
        tokens = tokens.to(device)
        return model.encode_text(tokens, cross_modal=True)


def image2feature(model, images, device):
    # return image2feature_clip(model, images, device)
    return model.encode_image(images, cross_modal=True)


if __name__ == '__main__':
    # clipModelName = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    processList = []
    fileList = list(os.walk("output/vitb32/coco"))
    for val in fileList:
        if "checkpoint_best.pth" in val[2]:
            processList.append(os.path.join(val[0], "checkpoint_best.pth"))
    for checkPointPath in processList:
        run(checkPointPath)
