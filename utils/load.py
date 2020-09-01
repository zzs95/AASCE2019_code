#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
# from PIL import Image
import cv2
import math
import random
import glob
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from .utils import resize_and_crop, get_square, normalize, hwc_to_chw


def get_ids(dir):
    """Returns a list of the ids in the directory"""
    data_list = glob.glob(dir+'*')
    return np.arange(len(data_list))


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i) for i in range(n) for id in ids)


def load_all_imgs(ids, dir, scale):
    """From a list of tuples, returns the correct cropped img"""
    imgs = []
    for id in ids:
        im = cv2.imread(dir[id], flags=0)
        im = resize_and_crop(im, scale)

        im = normalize(im)
        imgs.append(im)
    # imgs = torch.from_numpy(np.array(imgs)).cuda()
    return imgs

def load_all_label(ids, dir, scale):
    """From a list of labels, Generate a set of high light masks"""
    labels = []
    for id in ids:
        label = cv2.imread(dir[id], flags=0)
        label = resize_and_crop(label, scale)
        label = label.astype('float32')
        label = normalize(label)
        label = np.concatenate([label, label], axis=0)
        # label = torch.cat([label, label])
        label[1] = label.max() - label[1]
        labels.append(label)
    # labels = torch.from_numpy(np.array(labels)).cuda()
    return labels





dir_val_img = "E:\CSI2019\AASCE2019\data_stage1_secaled0.3/train_test\images\sunhl-1th-01-Mar-2017-310 a ap.jpg"
im = cv2.imread(dir_val_img)
orignal_shape = im.shape
del im

def get_scale_size(scale):
    w = orignal_shape[0]
    h = orignal_shape[1]
    newW = int(w * scale)
    if newW % 2 != 0:
        newW = newW + 1
    newH = int(h * scale)
    if newH % 2 != 0:
        newH = newH + 1
    return (newH, newW)

class ScaledDataset(Dataset):
    def __init__(self, ids, dir_img, dir_label, scale, gpu=False, data_aug=False):
        self.img_all = load_all_imgs(ids, sorted(dir_img), scale)
        self.label_all = load_all_label(ids, sorted(dir_label), scale)

        self.orignal_shape = orignal_shape
        self.gpu = gpu
        self.data_aug = data_aug

        
    def __len__(self):
        return len(self.img_all)
    
    def __getitem__(self, index):
        img = np.array(self.img_all[index], copy=True)
        label = np.array(self.label_all[index], copy=True)
        if self.data_aug:
            trans = data_augmentation()

            # img = trans.img_aug(img)
            # label[0] = trans.mask_aug(label[0])
            img, label[0] = trans.aug(img, label[0])

            label[0] = label[0]/(label[0].max()-label[0].min())+label[0].min()
            label[1] = 1 - label[0]
            label[1] = label[1]/(label[1].max()-label[1].min())+label[1].min()
            del trans
        img = img[0, 0:688, 0:254].reshape((1, 688, 254))
        label = label[:, 0:688, 0:254].reshape((2, 688, 254))
        return img, label

class predScaledDataset(Dataset):
    def __init__(self, ids, dir_img, scale, gpu=False, data_aug=False):
        self.img_all = load_all_imgs(ids, dir_img, scale)
        self.orignal_shape = orignal_shape
        self.gpu = gpu
        self.data_aug = data_aug
        
    def __len__(self):
        return len(self.img_all)
    
    def __getitem__(self, index):
        img = self.img_all[index]
        if self.data_aug:
            x = torch_data_augmentation()
            img = x.img_aug(img)
        # img = img[0, 0:688, 0:254].reshape((1, 688, 254))
        img = img[0, 0:688, 0:254].reshape((1, 688, 254))
        return img
        



class data_augmentation:
    def __init__(self):
        self.img_height = orignal_shape[0]
        self.img_height_scaled = int(1 * self.img_height)
        self.img_width = orignal_shape[1]
        self.img_width_scaled = int(1 * self.img_width)

        self.height, self.width = self.img_height_scaled, self.img_width_scaled
        self.center = (self.width / 2., self.height / 2.)
        # affrat data_augmentation
        self.affrat = random.uniform(0.9, 1.1)
        self.halfl_w = min(self.width - self.center[0], (self.width - self.center[0]) * self.affrat)
        self.halfl_h = min(self.height - self.center[1], (self.height - self.center[1]) * self.affrat)
        # rotated augmentation
        self.angle = random.uniform(0, 5)
        if random.randint(0, 1):
            self.angle *= -1
        self.rotMat = cv2.getRotationMatrix2D(self.center, self.angle, 1.0)
        self.flip = random.randint(0, 1)


    def aug(self, imgs, masks):
        imgs = imgs.squeeze()
        masks = masks.squeeze()
        # imgs = cv2.resize(imgs[int(self.center[1] - self.halfl_h): int(self.center[1] + self.halfl_h + 1),
        #                   int(self.center[0] - self.halfl_w): int(self.center[0] + self.halfl_w + 1)], (self.width, self.height))
        # imgs = cv2.warpAffine(imgs, self.rotMat, (self.width, self.height))

        # masks = cv2.resize(masks[int(self.center[1] - self.halfl_h): int(self.center[1] + self.halfl_h + 1),
        #                   int(self.center[0] - self.halfl_w): int(self.center[0] + self.halfl_w + 1)], (self.width, self.height))
        # masks = cv2.warpAffine(masks, self.rotMat, (self.width, self.height))
        if self.flip:
            imgs = cv2.flip(imgs, flipCode=1)
            masks = cv2.flip(masks, flipCode=1)
        else:
            imgs = imgs
            masks = masks
        imgs = imgs.reshape((1, self.height, self.width))
        masks = masks.reshape((1, self.height, self.width))
        return imgs, masks

    def img_aug(self, imgs):
        imgs = imgs.squeeze()
        # imgs = cv2.resize(imgs[int(self.center[1] - self.halfl_h): int(self.center[1] + self.halfl_h + 1),
        #                   int(self.center[0] - self.halfl_w): int(self.center[0] + self.halfl_w + 1)], (self.width, self.height))
        # imgs = cv2.warpAffine(imgs, self.rotMat, (self.width, self.height))
        if self.flip:
            imgs = cv2.flip(imgs, flipCode=1)
        imgs = imgs.reshape((1, self.height, self.width))
        return imgs

    def mask_aug(self, masks):
        masks = masks.squeeze()
        # imgs = cv2.resize(imgs[int(self.center[1] - self.halfl_h): int(self.center[1] + self.halfl_h + 1),
        #                   int(self.center[0] - self.halfl_w): int(self.center[0] + self.halfl_w + 1)], (self.width, self.height))
        # imgs = cv2.warpAffine(imgs, self.rotMat, (self.width, self.height))
        if self.flip:
            masks = cv2.flip(masks, flipCode=1)
        masks = masks.reshape((1, self.height, self.width))
        return masks

    def masks_label_aug(self, masks_and_label):
        masks = masks_and_label[0]
        labels = masks_and_label[1]
        for i in range(masks.shape[0]):
            mask = masks[i]
            mask = cv2.resize(mask[int(self.center[1] - self.halfl_h): int(self.center[1] + self.halfl_h + 1),
                              int(self.center[0] - self.halfl_w): int(self.center[0] + self.halfl_w + 1)], (self.width, self.height))
            if not i == 19:
                mask = cv2.warpAffine(mask, self.rotMat, (self.width, self.height))
            elif i==19:
                mask = cv2.warpAffine(mask, self.rotMat, (self.width, self.height), borderValue=1)

            masks[i] = mask

        for i in range(labels.__len__()):
            annot = labels
            x = annot[i][0]
            y = annot[i][1]
            x = (x - self.center[0]) / self.halfl_w * (self.width - self.center[0]) + self.center[0]
            y = (y - self.center[1]) / self.halfl_h * (self.height - self.center[1]) + self.center[1]
            if x >= 0 and y >= 0:
                R = self.rotMat[:, : 2]
                W = np.array([self.rotMat[0][2], self.rotMat[1][2]])
                annot[i] = np.dot(R, annot[i]) + W

        labels = annot
        masks_and_label = [masks, labels]
        return masks_and_label
    
import torchsample
from torchsample import transforms
class torch_data_augmentation:
    def __init__(self, ):
        '''
        hwc
        '''
        random.seed = 0
        self.img_height = orignal_shape[0]
        self.img_height_scaled = int(1 * self.img_height)
        self.img_width = orignal_shape[1]
        self.img_width_scaled = int(1 * self.img_width)

        self.height, self.width = self.img_height_scaled, self.img_width_scaled

        self.center = (self.width / 2., self.height / 2.)

        # rotation
        self.rotation_range = 2
        self.degree = random.uniform(-self.rotation_range, self.rotation_range)

        # translation
        self.translation_range = 0.1
        if isinstance(self.translation_range, float):
            translation_range = (self.translation_range, self.translation_range)
        self.height_range = translation_range[0]
        self.width_range = translation_range[1]
        self.random_height = random.uniform(-self.height_range, self.height_range)
        self.random_width = random.uniform(-self.width_range, self.width_range)

        # shear
        self.shear_range = 1
        self.shear = random.uniform(-self.shear_range, self.shear_range)

        # zoom
        self.zoom_range = (0.9, 1.3)
        self.zx = random.uniform(self.zoom_range[0], self.zoom_range[1])
        self.zy = random.uniform(self.zoom_range[0], self.zoom_range[1])

        # Gamma
        self.Gamma_range = (0.8, 1.1)
        self.Gammavalue = random.uniform(self.Gamma_range[0], self.Gamma_range[1])

        # Brightness
        self.bright_range = (-0.2, 0.2)
        self.brightvalue = random.uniform(self.bright_range[0], self.bright_range[1])

        # Saturation
        self.saturation_range = (-0.2, 0.2)
        self.saturationvalue = random.uniform(self.saturation_range[0], self.saturation_range[1])

        # Contrast
        self.Contrast_range = (0, 0.2)
        self.Contrastvalue = random.uniform(self.Contrast_range[0], self.Contrast_range[1])

        self.lazy = False

    def img_aug(self, imgs):
        imgs = imgs.cpu()
        self.transforms = [transforms.Rotate(self.degree, lazy=True),
                           transforms.Translate([self.random_height, self.random_width], lazy=True),
                           transforms.Shear(self.shear, lazy=True),
                           transforms.Zoom([self.zx, self.zy], lazy=True)]
        tform_matrix = self.transforms[0](imgs[0].unsqueeze(0))
        for tform in self.transforms[1:]:
            tform_matrix = tform_matrix.mm(tform(imgs[0].unsqueeze(0)))
        self.tform_matrix = tform_matrix
        for i, img in enumerate(imgs):
            img = transforms.Affine(tform_matrix, 'nearest')(img.unsqueeze(0))
            img = transforms.Brightness(self.brightvalue)(img.unsqueeze(0))
            img = transforms.Gamma(self.Gammavalue)(img.unsqueeze(0))
            # img = transforms.Contrast(self.Contrastvalue)(img.unsqueeze(0))
            img = img.reshape((self.height, self.width))
            imgs[i] = img
        return imgs.cuda()

    def masks_aug(self, masks):
        masks = masks.cpu()
        self.transforms = [transforms.Rotate(self.degree, lazy=True),
                           transforms.Translate([self.random_height, self.random_width], lazy=True),
                           transforms.Shear(self.shear, lazy=True),
                           transforms.Zoom([self.zx, self.zy], lazy=True)]
        tform_matrix = self.transforms[0](masks[0].unsqueeze(0))
        for tform in self.transforms[1:]:
            tform_matrix = tform_matrix.mm(tform(masks[0].unsqueeze(0)))
        self.tform_matrix = tform_matrix
        for i, mask in enumerate(masks):
            mask = transforms.Affine(tform_matrix)(mask.unsqueeze(0))
            mask = mask.reshape((self.height, self.width))
            masks[i] = mask
        return masks.cuda()

    def labels_aug(self, labels, imgs):
        labels = torch.FloatTensor(labels)
        self.transforms = [transforms.Rotate(self.degree, lazy=True),
                           transforms.Translate([self.random_height, self.random_width], lazy=True),
                           transforms.Shear(self.shear, lazy=True),
                           transforms.Zoom([self.zx, self.zy], lazy=True)]
        tform_matrix = self.transforms[0](imgs[0].unsqueeze(0))
        for tform in self.transforms[1:]:
            tform_matrix = tform_matrix.mm(tform(imgs[0].unsqueeze(0)))
        self.tform_matrix = tform_matrix

        self.halfl_w = self.width - self.center[0]
        self.halfl_h = self.height - self.center[1]
        annot = labels
        for i, label in enumerate(labels):
            x = label[0]
            y = label[1]
            x = (x - self.center[0]) / self.halfl_w * (self.width - self.center[0]) + self.center[0]
            y = (y - self.center[1]) / self.halfl_h * (self.height - self.center[1]) + self.center[1]
            if x >= 0 and y >= 0:
                R = self.tform_matrix[:, : 2][: 2]
                W = torch.FloatTensor([self.tform_matrix[0][2].item(), self.tform_matrix[1][2].item()])
                # annot_np = np.dot(R.numpy(), label.numpy()) + W.numpy()
                annot_np = torch.mm(R, label) + W
                annot[i] = torch.from_numpy(annot_np)
        labels = annot
        return labels
