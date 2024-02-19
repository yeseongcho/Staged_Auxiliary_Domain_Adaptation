import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils import data
from PIL import Image
from os.path import join
import scipy.misc as m
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class Pairedcityscapes(data.Dataset):
    colors = [  
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    def __init__(self, fs_root, cs_root, fs_list_path, cs_list_path, max_iters=None, mean=(128, 128, 128), ignore_label=255, set='val'):
        self.fs_root = fs_root
        self.cs_root = cs_root
        self.fs_list_path = fs_list_path
        self.cs_list_path = cs_list_path
        self.ignore_label = ignore_label
        self.mean = mean
        self.img_ids = [i_id.strip() for i_id in open(fs_list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(19)))
        for name in self.img_ids:
            cs_img_file = osp.join(self.cs_root, "leftImg8bit/%s/%s" % (self.set, name[:-21]+'.png'))
            fs_img_file = osp.join(self.fs_root, "leftImg8bit_foggy/%s/%s" % (self.set, name)) ## SF
            label_file = osp.join(self.cs_root, "gtFine/%s/%s" % (self.set, name[:-32]+'gtFine_labelIds.png')) 
            self.files.append({
                "fs_img": fs_img_file,
                "cs_img": cs_img_file,
                "label": label_file,
                "cs_name": name[:-21]+'.png',
                "fs_name": name
            })



    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        datafiles = self.files[index]
      
        fs_image = Image.open(datafiles["fs_img"]).convert('RGB')
        cs_image = Image.open(datafiles["cs_img"]).convert('RGB')
        fs_arry = np.array(fs_image)
        cs_arry = np.array(cs_image)
        label = Image.open(datafiles["label"])
        fs_name = datafiles["fs_name"]
        cs_name = datafiles["cs_name"]
        # resize
        w, h = fs_image.size
        fs_image, cs_image, label = self._apply_transform(fs_image, cs_image, label, scale=0.8)

        crop_size = min(500, min(fs_image.size[:2]))
        i, j, h, w = transforms.RandomCrop.get_params(fs_image, output_size=(crop_size,crop_size)) 
        fs_image = TF.crop(fs_image, i, j, h, w) 
        cs_image = TF.crop(cs_image, i, j, h, w)
        label = TF.crop(label, i, j, h, w)

        if random.random() > 0.5:
            fs_image = TF.hflip(fs_image)
            cs_image = TF.hflip(cs_image)
            label = TF.hflip(label)

        fs_image = np.asarray(fs_image, np.float32)
        cs_image = np.asarray(cs_image, np.float32)
        label = self.encode_segmap(np.array(label, dtype=np.float32))

        classes = np.unique(label)
        lbl = label.astype(float)

        label = lbl.astype(int)

        size = fs_image.shape
        fs_image = fs_image[:, :, ::-1]  # change to BGR
        fs_image -= self.mean
        fs_image = fs_image.transpose((2, 0, 1))
        cs_image = cs_image[:, :, ::-1]  # change to BGR
        cs_image -= self.mean
        cs_image = cs_image.transpose((2, 0, 1))

        return fs_image.copy(), cs_image.copy(),label.copy(), np.array(size), fs_name, cs_name, fs_arry, cs_arry
    
    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def _apply_transform(self, img1, img2, lbl, scale=(0.7, 1.3), crop_size=600):
        (W, H) = img1.size[:2]
        if isinstance(scale, tuple):
            scale = random.random() * 0.6 + 0.7

        tsfrms = []
        tsfrms.append(transforms.Resize((int(H * scale), int(W * scale))))
        tsfrms = transforms.Compose(tsfrms)

        return tsfrms(img1), tsfrms(img2), tsfrms(lbl)

