import os

import numpy as np
import torch
from PIL import Image
import cv2
import math
import glob
import random
import collections
from .utils import resize_and_crop, get_square, normalize, hwc_to_chw
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def gennerate_4point_masks(labels, size):
    """From a list of labels, Generate a set of high light masks"""
    labels = labels.reshape((-1, 2))
    masks_channels = int(labels.shape[0] / 4)
    # labels = labels.type(torch.FloatTensor)
    # labels = labels.to('cuda:0')
    # print('ok')

    # 20th masks
    masks = torch.zeros((masks_channels + 1, size[0], size[1]))
    # masks = masks.to('cuda:0')
    masks_temp = torch.zeros(size)
    for i, label in enumerate(labels):
        masks_temp += label_2_mask(label, size, normal_dist)
        if (i+1) % 4 == 0:
            masks[int((i+1)/4-1)] = masks_temp
            masks_temp = torch.zeros(size)

    masks[-1] = torch.ones(size) - masks.sum(0)
    masks[masks_channels] = masks[masks_channels] * ((masks[masks_channels] >= 0).type(torch.cuda.FloatTensor)) # 令小于0的元素等于零
    return masks

def gennerate_point_masks(labels, size):
    """From a list of labels, Generate a set of high light masks"""
    labels = labels.reshape((-1, 2))
    masks_channels = int(labels.shape[0])
    # labels = labels.type(torch.FloatTensor)
    # labels = labels.to('cuda:0')
    # print('ok')

    # 20th masks
    masks = np.zeros((masks_channels + 1, size[0], size[1]))
    for i, label in enumerate(labels):
        masks[i] = label_2_mask(label, size, normal_dist)

    masks[-1] = np.ones(size) - masks.sum(0)
    masks[masks_channels] = masks[masks_channels] * (masks[masks_channels] >= 0) # 令小于0的元素等于零
    return masks

def gennerate_segm_masks(labels, size):
    """From a list of labels, Generate a set of high light masks"""
    labels = labels.reshape((-1, 2))
    masks_channels = int(labels.shape[0] / 4)
    # labels = labels.type(torch.FloatTensor)
    # labels = labels.to('cuda:0')
    # print('ok')

    # 20th masks
    # masks = np.zeros((masks_channels + 1, size[0], size[1]))
    masks = np.zeros((masks_channels+1, size[0], size[1]))
    # masks = masks.to('cuda:0')
    mask_temp = np.zeros(size)
    labels = labels.reshape(-1, 4, 2)
    for i, label in enumerate(labels):
        label_temp_1 = np.array(label[-2], copy=True)
        label[-2] = label[-1]
        label[-1] = label_temp_1
        labels_ = np.array(label).reshape((-1, 1, 2)).astype(np.int32)
        masks[i] = np.array(cv2.drawContours(mask_temp, [labels_], -1, 1,  cv2.FILLED)).astype(np.float)
        mask_temp = np.zeros(size)

    masks[-1] = np.ones(size) - masks.sum(0)
    # masks[masks_channels] = masks[masks_channels] * ((masks[masks_channels] >= 0)) # 令小于0的元素等于零
    return masks

def label_2_mask(label_scaled, img_size_scaled, normal_dist):
    # label_scale = label_scaled.type(torch.cuda.IntTensor)
    label_scale = label_scaled.astype(int)
    normal_dist_width = int(normal_dist.shape[0])
    normal_dist_half_width = int(normal_dist.shape[0]/2)

    img_width_scaled = int(img_size_scaled[1])
    img_height_scaled = int(img_size_scaled[0])
    extend_width = int(img_width_scaled + normal_dist_width)
    extend_height = int(img_height_scaled + normal_dist_width)
    mask_extend = np.zeros((extend_height, extend_width))

    nor_x_begin = int(label_scale[1])
    nor_x_end = int(label_scale[1] + normal_dist_width)
    nor_y_begin = int(label_scale[0])
    nor_y_end = int(label_scale[0] + normal_dist_width)

    mask_extend[nor_x_begin:nor_x_end, nor_y_begin:nor_y_end] = normal_dist
    mask_begin = int(normal_dist_half_width)
    mask_width_end = int(mask_begin+img_width_scaled)
    mask_height_end = int(mask_begin+img_height_scaled)


    mask = mask_extend[mask_begin:mask_height_end, mask_begin:mask_width_end]
    return mask

def normal_distribution(size=400):
    res = 6/size
    x = np.arange(-3, 3, res)
    y = np.arange(-3, 3, res)
    x, y = np.meshgrid(x, y)
    # z = (1 / (2 * math.pi * 1 ** 2)) * np.exp(-(x ** 2 + y ** 2) / 2 * 1 ** 2)
    z = (1 / (2 * math.pi * 1 ** 2)) * np.exp(-(x ** 2 + y ** 2) / 2 * 1 ** 2)
    z = ((z / z.max()))
    # z = torch.from_numpy(z).type(torch.cuda.FloatTensor)
    return z

def get_masks(labels, size):

    masks = gennerate_segm_masks(labels, size)
    # masks = gennerate_point_masks(labels, size)
    # masks = gennerate_point_masks()

    return masks

normal_dist = normal_distribution(40)
def region_crop(img, anchor, masks, crop_size):
    img_shape = np.array(img.shape)

    if not crop_size.size == 2:
        crop_size = np.array([crop_size, crop_size])
    margin = int(crop_size.max())
    crop_x = int(crop_size[0])
    crop_y = int(crop_size[1])
    half_crop_x = int(crop_x * 0.5)
    half_crop_y = int(crop_y * 0.5)

    big_img_shape = np.array(img_shape + 2 * margin).astype(int)

    cropbegin_x0 = int(anchor[0] - half_crop_x)
    cropbegin_x0_maigin = int(cropbegin_x0 + margin)
    cropbegin_y0 = int(anchor[1] - half_crop_y)
    cropbegin_y0_maigin = int(cropbegin_y0 + margin)
    begin_crop = np.array((cropbegin_x0, cropbegin_y0))
    cropend_x0_maigin = int(cropbegin_x0_maigin + crop_x)
    cropend_y0_maigin = int(cropbegin_y0_maigin + crop_y)

    imgbig = np.zeros(big_img_shape)
    imgbig[margin: margin+img_shape[0], margin: margin+img_shape[1]] = img
    img_crop = imgbig[cropbegin_y0_maigin:cropend_y0_maigin, cropbegin_x0_maigin:cropend_x0_maigin]
    img_crop = img_crop.reshape((1, crop_y, crop_x))
    channels_num = masks.shape[0]
    masksbig = np.zeros((channels_num, big_img_shape[0], big_img_shape[1]))
    masksbig[:, margin:margin+img_shape[0], margin:margin+img_shape[1]] = masks
    masks_crop = masksbig[:, cropbegin_y0_maigin:cropend_y0_maigin, cropbegin_x0_maigin:cropend_x0_maigin]



    # return img_crop, masks_crop, begin_crop

    # channel squeeze
    # non_zero_index = masks_crop.sum([1, 2]) > 0
    # masks_sq = masks_crop[non_zero_index]
    # if masks_sq.shape[0] > 8:
    #     # print('down'+str(masks_sq.shape[0]))
    #     while masks_sq.shape[0] > 8:
    #         non_zero_index = masks_sq.sum([1, 2]) > 0
    #         min_indx = masks_sq.sum([1, 2]).min(0)[1].item()
    #         non_zero_index[min_indx] = 0
    #         masks_sq = masks_sq[non_zero_index]
    # elif masks_sq.shape[0] < 8:
    #     # print('up'+str(masks_sq.shape[0]))
    #     while masks_sq.shape[0] < 8:
    #         mask_zero = np.zeros(masks_sq[0].shape).expand_dim(dim=0)
    #         masks_sq = np.concatenate([masks_sq, mask_zero])
    # else:
    #     # print('stay' + str(masks_sq.shape[0]))

    return img_crop, masks_crop, begin_crop