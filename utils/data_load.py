#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
from PIL import Image
import cv2
import math
import random
import glob
import torch
from scipy.io import loadmat

from .utils import resize_and_crop, get_square, normalize, hwc_to_chw

def load_all_image(path):
    print('loading all image')
    images_all_list = glob.glob(path+'images\*.jpg')
    images_all = load_images(images_all_list)
    images = np.array(images_all)
    return images

def load_all_seg_mask(path):
    print('loading all seg_mask')
    seg_masks_all_list = glob.glob(path+'seg_mask\*.jpg')
    seg_masks_all = load_images(seg_masks_all_list)
    seg_masks = np.array(seg_masks_all)
    return seg_masks

def load_all_labels(path, type='mat'):
    print('loading all labels')
    if type == 'mat':
        labels_all_list = glob.glob(path+'labels\*.mat')
        labels_all = load_labels_mat(labels_all_list)

    elif type == 'npy':
        labels_all_list = glob.glob(path+'labels\*.npy')
        labels_all = load_labels_npy(labels_all_list)
    else:
        print('type error')
        labels_all = []
    return labels_all


def load_all_data_from_npy(path):
    images_all_list = glob.glob(path+'images\*.jpg')
    images_all = load_images(images_all_list)
    labels_all_list = glob.glob(path+'labels\*.npy')
    labels_all = load_labels_npy(labels_all_list)
    seg_masks_all_list = glob.glob(path+'seg_mask\*.jpg')
    seg_masks_all = load_images(seg_masks_all_list)
    if not (len(images_all) == len(labels_all) and len(seg_masks_all) == len(labels_all)):
        assert 'data error'
    labels = np.array(labels_all)

    # images = np.array(images_all)
    # seg_masks = np.array(seg_masks_all)

    images = torch.FloatTensor(images_all)
    seg_masks = torch.FloatTensor(seg_masks_all)

    # images = torch.FloatTensor(images).type(torch.cuda.FloatTensor)
    # seg_masks = torch.FloatTensor(seg_masks).type(torch.cuda.FloatTensor)
    return images, labels, seg_masks

def load_images(images_list):
    images = []
    for image_dir in images_list:
        img = pil_loader(image_dir)
        if img.shape[1] % 2:
            width = img.shape[1]-1
        else:
            width = img.shape[1]
        if img.shape[0] % 2:
            height = img.shape[0]-1
        else:
            height = img.shape[0]


        images.append(img[0:height, 0:width])
    return images

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return np.array(img.convert('L'))

def load_labels_npy(dir_list):
    labels = []
    for label_dir in dir_list:
        # labels.append(loadmat(label_dir)['p2'])
        labels.append(np.load(label_dir))
    return labels

def load_labels_mat(dir_list):
    labels = []
    for label_dir in dir_list:
        labels.append(loadmat(label_dir)['p2'])
        # labels.append(np.load(label_dir))
    return labels


class data_augmentation:
    def __init__(self, orignal_shape):
        self.img_height = orignal_shape[0]
        self.img_height_scaled = int(1 * self.img_height)
        self.img_width = orignal_shape[1]
        self.img_width_scaled = int(1 * self.img_width)

        self.height, self.width = self.img_height_scaled, self.img_width_scaled
        self.center = (self.width / 2., self.height / 2.)
        # affrat data_augmentation
        self.affrat = random.uniform(0.7, 1.1)
        self.halfl_w = min(self.width - self.center[0], (self.width - self.center[0]) * self.affrat)
        self.halfl_h = min(self.height - self.center[1], (self.height - self.center[1]) * self.affrat)
        # rotated augmentation
        self.angle = random.uniform(0, 5)
        if random.randint(0, 1):
            self.angle *= -1
        self.rotMat = cv2.getRotationMatrix2D(self.center, self.angle, 1.0)
        self.flip = random.randint(0, 1)



    def img_aug(self, imgs):
        for i, img in enumerate(imgs):
            img = img.squeeze()
            img = cv2.resize(img[int(self.center[1] - self.halfl_h): int(self.center[1] + self.halfl_h + 1),
                              int(self.center[0] - self.halfl_w): int(self.center[0] + self.halfl_w + 1)], (self.width, self.height))
            img = cv2.warpAffine(img, self.rotMat, (self.width, self.height), borderValue=0)
            if self.flip:
                img = cv2.flip(img, flipCode=1)
            img = img.reshape((1, self.height, self.width))
            imgs[i] = np.array(img).squeeze()
        return imgs

    def masks_aug(self, masks):

        for i in range(masks.shape[0]):
            mask = masks[i].squeeze()
            mask = cv2.resize(mask[int(self.center[1] - self.halfl_h): int(self.center[1] + self.halfl_h + 1),
                              int(self.center[0] - self.halfl_w): int(self.center[0] + self.halfl_w + 1)], (self.width, self.height))
            if not i == masks.shape[0]-1:
                mask = cv2.warpAffine(mask, self.rotMat, (self.width, self.height), borderValue=0)
            elif i == masks.shape[0]-1:
                mask = cv2.warpAffine(mask, self.rotMat, (self.width, self.height), borderValue=1)
            if self.flip:
                mask = cv2.flip(mask, flipCode=1)
            mask = mask.reshape((1, self.height, self.width))
            masks[i] = mask
        return masks

    def labels_aug(self, labels):

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
        return labels
#
# import torchsample
# from torchsample import transforms
# class torch_data_augmentation:
#     def __init__(self, ):
#         random.seed = 0
#         self.img_width = 200
#         self.img_width_scaled = int(1 * self.img_width)
#         self.height, self.width = self.img_width_scaled, self.img_width_scaled
#         self.center = (self.width / 2., self.height / 2.)
#
#         # rotation
#         self.rotation_range = 10
#         self.degree = random.uniform(-self.rotation_range, self.rotation_range)
#
#         # translation
#         self.translation_range = 0.8
#         if isinstance(self.translation_range, float):
#             translation_range = (self.translation_range, self.translation_range)
#         self.height_range = translation_range[0]
#         self.width_range = translation_range[1]
#         self.random_height = random.uniform(-self.height_range, self.height_range)
#         self.random_width = random.uniform(-self.width_range, self.width_range)
#
#         # shear
#         self.shear_range = 5
#         self.shear = random.uniform(-self.shear_range, self.shear_range)
#
#         # zoom
#         self.zoom_range = (0.9, 1.3)
#         self.zx = random.uniform(self.zoom_range[0], self.zoom_range[1])
#         self.zy = random.uniform(self.zoom_range[0], self.zoom_range[1])
#
#         # Gamma
#         self.Gamma_range = (0.3, 1.8)
#         self.Gammavalue = random.uniform(self.Gamma_range[0], self.Gamma_range[1])
#
#         # Brightness
#         self.bright_range = (-0.2, 0.2)
#         self.brightvalue = random.uniform(self.bright_range[0], self.bright_range[1])
#
#         # Saturation
#         self.saturation_range = (-0.3, 0.3)
#         self.saturationvalue = random.uniform(self.saturation_range[0], self.saturation_range[1])
#
#         # Contrast
#         self.Contrast_range = (0.5, 1.5)
#         self.Contrastvalue = random.uniform(self.Contrast_range[0], self.Contrast_range[1])
#
#         self.lazy = False
#
#     def img_aug(self, imgs):
#         imgs = imgs.cpu()
#         self.transforms = [transforms.Rotate(self.degree, lazy=True),
#                            transforms.Translate([self.random_height, self.random_width], lazy=True),
#                            transforms.Shear(self.shear, lazy=True),
#                            transforms.Zoom([self.zx, self.zy], lazy=True)]
#         tform_matrix = self.transforms[0](imgs[0].unsqueeze(0))
#         for tform in self.transforms[1:]:
#             tform_matrix = tform_matrix.mm(tform(imgs[0].unsqueeze(0)))
#         self.tform_matrix = tform_matrix
#         for i, img in enumerate(imgs):
#             img = transforms.Affine(tform_matrix)(img.unsqueeze(0))
#             img = transforms.Brightness(self.brightvalue)(img.unsqueeze(0))
#             img = transforms.Gamma(self.Gammavalue)(img.unsqueeze(0))
#             img = transforms.Contrast(self.Contrastvalue)(img.unsqueeze(0))
#             img = img.reshape((self.height, self.width))
#             imgs[i] = img
#         return imgs.cuda()
#
#     def masks_aug(self, masks):
#         masks = masks.cpu()
#         self.transforms = [transforms.Rotate(self.degree, lazy=True),
#                            transforms.Translate([self.random_height, self.random_width], lazy=True),
#                            transforms.Shear(self.shear, lazy=True),
#                            transforms.Zoom([self.zx, self.zy], lazy=True)]
#         tform_matrix = self.transforms[0](masks[0].unsqueeze(0))
#         for tform in self.transforms[1:]:
#             tform_matrix = tform_matrix.mm(tform(masks[0].unsqueeze(0)))
#         self.tform_matrix = tform_matrix
#         for i, mask in enumerate(masks):
#             mask = transforms.Affine(tform_matrix)(mask.unsqueeze(0))
#             mask = mask.reshape((self.height, self.width))
#             masks[i] = mask
#         return masks.cuda()
#
#     def labels_aug(self, labels, imgs):
#         labels = torch.FloatTensor(labels)
#         self.transforms = [transforms.Rotate(self.degree, lazy=True),
#                            transforms.Translate([self.random_height, self.random_width], lazy=True),
#                            transforms.Shear(self.shear, lazy=True),
#                            transforms.Zoom([self.zx, self.zy], lazy=True)]
#         tform_matrix = self.transforms[0](imgs[0].unsqueeze(0))
#         for tform in self.transforms[1:]:
#             tform_matrix = tform_matrix.mm(tform(imgs[0].unsqueeze(0)))
#         self.tform_matrix = tform_matrix
#
#         self.halfl_w = self.width - self.center[0]
#         self.halfl_h = self.height - self.center[1]
#         annot = labels
#         for i, label in enumerate(labels):
#             x = label[0]
#             y = label[1]
#             x = (x - self.center[0]) / self.halfl_w * (self.width - self.center[0]) + self.center[0]
#             y = (y - self.center[1]) / self.halfl_h * (self.height - self.center[1]) + self.center[1]
#             if x >= 0 and y >= 0:
#                 R = self.tform_matrix[:, : 2][: 2]
#                 W = torch.FloatTensor([self.tform_matrix[0][2].item(), self.tform_matrix[1][2].item()])
#                 # annot_np = np.dot(R.numpy(), label.numpy()) + W.numpy()
#                 annot_np = torch.mm(R, label) + W
#                 annot[i] = torch.from_numpy(annot_np)
#         labels = annot
#         return labels
