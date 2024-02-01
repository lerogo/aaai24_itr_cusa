import json
import os
import traceback

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from clip import clip
from .utils import pre_caption


class cross_coco_dataset(Dataset):
    def __init__(self, root, transform=None, split="train", max_words=64):
        self.root = root
        self.transform = transform
        self.split = split
        self.max_words = max_words

        self.dataPath = os.path.join(self.root, "new_{}.json".format(self.split))
        with open(self.dataPath, "r", encoding="utf8") as f:
            self.dataList = json.load(f)

        self.img_ids = {}
        n = 0
        for ann in self.dataList:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

        if self.split == "experiment":
            self.split = "train"
        try:
            self.unicom_fea = np.load(os.path.join(self.root, "{}_unicom.npy".format(self.split)), allow_pickle=True).item()
        except Exception as e:
            traceback.print_exc()
            self.unicom_fea = None

    def __len__(self):
        return len(self.dataList)

    def __getitem__(self, index):
        tmpData = self.dataList[index]

        caption = pre_caption(tmpData["caption"], self.max_words)

        raw_caption = caption

        image_feature = torch.tensor([0.0])
        if self.unicom_fea is not None:
            image_feature = self.unicom_fea.get(tmpData["image_id"])[0]

        caption = clip.tokenize(caption)[0]

        im = Image.open(os.path.join(self.root, tmpData["image_path"])).convert('RGB')
        im = self.transform(im)

        return im, caption, image_feature, raw_caption, self.img_ids[tmpData["image_id"]]


class cross_coco_test_dataset(Dataset):
    def __init__(self, root, transform=None, split="test", max_words=64):
        self.root = root
        self.transform = transform
        self.split = split
        self.max_words = max_words
        self.dataPath = os.path.join(self.root, "new_{}.json".format(self.split))

        with open(self.dataPath, "r", encoding="utf8") as f:
            """
            [{
                "image_path": "COCO_val2014_000000184613.jpg",
                "image_id": "184613",
                "caption": "A young man holding an umbrella next to a herd of cattle ."
            }, ...]
            """
            self.dataList = json.load(f)
        """
        {
            "<image_id>":{
                "image_path": "COCO_val2014_000000184613.jpg",
                "caption":[//5 captions]
            }
        }
        """
        tmpData = {}
        for val in self.dataList:
            if val.get("image_id") not in tmpData:
                tmpData[val.get("image_id")] = {
                    "image_path": val.get("image_path"), "caption": [pre_caption(val.get("caption"), self.max_words)]}
            else:
                tmpData[val.get("image_id")]["caption"].append(pre_caption(val.get("caption"), self.max_words))
        # sort image_id keys to keep the order of images
        imgIdKeys = sorted(list(tmpData.keys()))
        self.text = []
        self.image = []
        self.img2txt = {}
        self.txt2img = {}
        txt_id = 0
        for id, key in enumerate(imgIdKeys):
            self.image.append(tmpData[key]["image_path"])
            self.img2txt[id] = []
            for tid, caption in enumerate(tmpData[key]["caption"]):
                self.text.append(caption)
                self.img2txt[id].append(txt_id)
                self.txt2img[txt_id] = id
                txt_id += 1

    def preprocess_text(self, textList):
        preCaptionList = clip.tokenize(textList)
        return preCaptionList

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):

        im = Image.open(os.path.join(self.root, self.image[index])).convert('RGB')
        im = self.transform(im)

        return im, index
