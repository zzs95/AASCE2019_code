from functools import partial
import numpy as np

from pathlib import Path
import glob
import os
import torch
import random
from torch.utils.data import DataLoader, Dataset

from utils.preprocess import minmax_normalize, meanstd_normalize, padding
from utils import load_all_data_from_npy, get_masks, data_augmentation, \
    region_crop, load_all_image, load_all_labels, resize_and_crop
from utils import labels_split, fit_line


torch.set_default_tensor_type('torch.cuda.FloatTensor')



class TrainORIDataset(Dataset):
    def __init__(self, data_path):
        self.img_all = load_all_image(data_path)
        self.label_all = load_all_labels(data_path, 'mat')
        self.img_size = np.array(self.img_all[0].shape)
        self.data_type = 'Train'

    def __len__(self):
        return len(self.img_all)

    def __getitem__(self, index):
        img = np.array(self.img_all[index], copy=True)
        self.img_size = img.shape[0:2]
        # seg_mask = np.array(self.segmask_all[index], copy=True)
        labels = np.array(self.label_all[index], copy=True)
        labels1 = np.array(self.label_all[index], copy=True)
        masks = get_masks(labels1, self.img_size)

        img_region = np.expand_dims(img, 0)
        masks_region = masks
        img_region = img_region / img_region.max()

        return np.array(img_region), np.array(masks_region), np.array(labels)

    def _load_all_masks(self, label_all, img_size):
        print('preparing all masks data.....')
        masks = []
        for labels in label_all:
            mask = get_masks(labels, img_size)
            masks.append(mask)
        return masks

class PredORIDataset(Dataset):
    def __init__(self, data_path):
        self.images_all_list = glob.glob(data_path + 'images\*.jpg')
        self.img_all = load_all_image(data_path)
        self.img_size = np.array(self.img_all[0].shape)
        self.data_type = 'Pred'

    def __len__(self):
        return len(self.img_all)

    def __getitem__(self, index):
        img = np.array(self.img_all[index])
        self.img_size = img.shape[0:2]

        img_region = np.expand_dims(img, 0)
        img_region = img_region / img_region.max()
        filename = os.path.split(self.images_all_list[index])[-1]
        return np.array(img_region), filename
