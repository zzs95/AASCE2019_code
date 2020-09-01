import sys
import pickle
# sys.path
import os
import glob
from optparse import OptionParser
import numpy as np
from pathlib import Path
from logger.log import debug_logger
# from logger.plot import history_ploter
# from loss.dice_loss import dice_coeff
# from loss.focal_loss import FocalLoss
# from loss.BCELoss2D import BCELoss2d

import cv2
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from stage2_pred_maskrcnn import stage2_maskrcnn

from dataset import TrainORIDataset, PredORIDataset
from model import UNet
from model import ResUNet
from utils import masks_2_labels_maximum, masks_2_labels_max_tread, masks_2_labels_thread, plot_img_and_label
torch.set_default_tensor_type('torch.cuda.FloatTensor')

class pred():
    def __init__(self, s1_modelw, s1_models, s2_model, stage3_boxmodel,stage3_pointmodel, dataLoader, output_dir,):

        self.dataset = dataLoader.dataset


        if self.dataset.data_type == 'Train':
            self.load_img = self._train_img
            logger.info('Train data is predicting.')
        elif self.dataset.data_type == 'Pred':
            self.load_img = self._pred_img
            logger.info('Pred data is predicting.')
        else:
            print("Dataset error.")

        self.standard_size = np.array((2300, 850))
        self.stage1W_size = np.array((690, 255))
        # self.stage1W_size = np.array((688, 254))
        self.stage1S_size = np.array((688, 254))
        self.stage2_size = np.array((806, 298))
        self.stage3_size = np.array((255, 255))

        self.pred_s1_WLM = stage_pred_function(s1_modelw, self.stage1W_size)
        self.pred_s1_SLM = stage_pred_function(s1_models, self.stage1S_size)
        self.pred_s2_BM = stage_pred_function(s2_model, self.stage2_size)
        self.pred_s2_maskrcnn = stage2_maskrcnn()
        self.pred_s3 = pred_stage3_refineBox(stage3_boxmodel, stage3_pointmodel, self.stage3_size)
    def _train_img(self, data_pack):
        return data_pack[0]
    def _pred_img(self, data_pack):
        return data_pack

    def forward(self):
        for data_id in range(98):
            print(data_id)

            img_ORI, filename = self.load_img(self.dataset[data_id])

            wholeLineMask1 = self.pred_s1_WLM.forward(img_ORI)
            img_WLM = np.array(wholeLineMask1[0] * img_ORI[0]).reshape(img_ORI.shape)
            det_full_masks, det_bbox, bbox_center, det_id = self.pred_s2_maskrcnn.forward(img_WLM, data_id, False)


            boxMask1 = self.pred_s2_BM.forward(img_WLM)
            segLineMask1 = self.pred_s1_SLM.forward(img_WLM)


            img_name = str(filename)
            print(img_name)
            wholeLineMask1 = self.pred_s1_WLM.forward(img_ORI)
            img_WLM = np.array(wholeLineMask1[0] * img_ORI[0]).reshape(img_ORI.shape)
            det_full_masks, det_bbox, bbox_center, det_id = self.pred_s2_maskrcnn.forward(img_WLM, data_id, False)

            boxMask1 = self.pred_s2_BM.forward(img_ORI)
            # bm_label = mask2_label(boxMask1, wholeLineMask1)
            # rough_label = label_distance_filter(bm_label, distance=(2000))
            # label_2 = np.concatenate([bbox_center, rough_label])
            # label_2, _ = outlier_filter(label_2)
            # line = fit_line_segment(label_2)

            refine_box = self.pred_s3.forward(img_ORI, det_bbox, bbox_center, det_full_masks, boxMask1)

            line = fit_line(bbox_center)
            line_refine_box = find_line_box(line, refine_box)
            line_refine_box = find_line_label_fix(line)

            line = fit_line_segment(bbox_center)
            line_refine_box = find_line_label_fix(line)

            print(data_id)
            img_ORI, filename = self.load_img(self.dataset[data_id])
            img_name = str(filename)
            print(img_name)
            wholeLineMask1 = self.pred_s1_WLM.forward(img_ORI)
            img_WLM = np.array(wholeLineMask1[0] * img_ORI[0]).reshape(img_ORI.shape)
            det_full_masks, det_bbox, bbox_center, det_id = self.pred_s2_maskrcnn.forward(img_WLM, data_id, False)

            boxMask1 = self.pred_s2_BM.forward(img_ORI)
            bm_label = mask2_label(boxMask1, wholeLineMask1)
            rough_label = label_distance_filter(bm_label, distance=(2000))
            label_2 = np.concatenate([bbox_center, rough_label])
            label_2, _ = outlier_filter(label_2)
            line = fit_line_segment(label_2)
            label_2f = line_outlier_filter(line, label_2)
            line = fit_line_segment(label_2f)

            line_refine_box = find_line_label_fix(line)
            np.save("./labels_pred/" + img_name + ".npy", line_refine_box)

            # return 0

    dedicate = [(1,), (2,),()]

    def pred_label(self):
        for data_id in range(98):
            img_ORI, filename = self.load_img(self.dataset[data_id])
            img_name = str(filename) + '.jpg'
            print(data_id)
            img_ORI, img_name = self.load_img(self.dataset[data_id])
            wholeLineMask1 = self.pred_s1_WLM.forward(img_ORI)
            img_WLM = np.array(wholeLineMask1[0] * img_ORI[0]).reshape(img_ORI.shape)
            det_full_masks, det_bbox, bbox_center, det_id = self.pred_s2_maskrcnn.forward(img_WLM, data_id, False)
            labels_rb = self.pred_s3_RB.forward(img_ORI, det_bbox, bbox_center, det_full_masks)
            np.save("E:\CSI2019\AASCE_stage4_Process\data\selected2\labels/" + img_name + ".npy", labels_rb)

    # for data_id in range(98):
#
#     img_ORI, filename = self.load_img(self.dataset[data_id])
#     img_name = str(filename) + '.jpg'
#     wholeLineMask1 = self.pred_s1_WLM.forward(img_ORI)
#     segmLineMask1 = self.pred_s1_SLM.forward(img_ORI)
#     # img_WLM = np.array(wholeLineMask1[0]*img_ORI[0]).reshape(img_ORI.shape)
#     boxMask1 = self.pred_s2_BM.forward(img_ORI)
#     bm_label = mask2_label(boxMask1, segmLineMask1)
#     correct_label = recorrect_label(bm_label, dif=500)
#     anchor, target_size = region_estimate(correct_label)
#     crop_size = region_expand(target_size, exp=0.)
#     img_crop, bc = img_region_crop(img_ORI, anchor, crop_size)
#     m_new = Image.fromarray(img_crop.squeeze() * 255)
#     m_new = m_new.convert('L')
#     m_new.save('E:\CSI2019\AASCE_stage4_Process\pred1/' + img_name)

    def crop_img(self):
        for data_id in range(98):
            print(data_id)
            img_ORI, filename = self.load_img(self.dataset[data_id])
            img_name = str(filename) + '.jpg'
            for i in range(10):
                img_ORI_shape = img_ORI.shape
                min_width = 3000
                if img_ORI_shape[-1] < min_width:
                    img_ORI = resize_and_crop(img_ORI,
                                              (int(img_ORI_shape[1] / img_ORI_shape[2] * min_width), int(min_width)))
                segmLineMask1 = self.pred_s1_SLM.forward(img_ORI)
                # wholeLineMask1 = self.pred_s1_WLM.forward(img_ORI)
                # img_WLM = np.array(wholeLineMask1[0]*img_ORI[0]).reshape(img_ORI.shape)
                boxMask1 = self.pred_s2_BM.forward(img_ORI)
                bm_label = mask2_label(boxMask1, segmLineMask1)
                rough_label = label_distance_filter(bm_label, distance=(2000+i*100))
                conf = confident(rough_label)

                if conf == 1.1:
                    anchor, target_size = region_estimate(rough_label)
                    crop_size = region_expand(target_size, exp=1 - conf)
                else:
                    anchor, crop_size = Boundary_stripping(rough_label, img_ORI.shape, strip=(1000 ))
                    # crop_size = region_expand(target_size, exp=1-conf)
                crop_size = even_size(crop_size)
                img_crop, begin_crop = img_region_crop(img_ORI, anchor, crop_size)
                img_ORI = img_crop
                print(conf)
            m_new = Image.fromarray(img_crop.squeeze() * 255)
            m_new = m_new.convert('L')
            m_new.save('E:\CSI2019\AASCE_stage4_Process\pred1/' + img_name)

    def pred_boxMask(self):
        for data_id in range(98):
            print(data_id)
            img_ORI, filename = self.load_img(self.dataset[data_id])
            img_name = str(filename) + '.jpg'
            # for i in range(10):
            # wholeLineMask1 = self.pred_s1_WLM.forward(img_ORI)
            # img_WLM = np.array(wholeLineMask1[0]*img_ORI[0]).reshape(img_ORI.shape)
            boxMask1 = self.pred_s2_BM.forward(img_ORI)

            m_new = Image.fromarray(boxMask1[-1].squeeze() * 255)
            m_new = m_new.convert('L')
            m_new.save('E:\CSI2019\AASCE_stage4_Process\pred2/' + img_name)

    def pred_maskrcnn(self):
        for data_id in range(98):
            print(data_id)
            img_ORI, img_name = self.load_img(self.dataset[data_id])
            wholeLineMask1 = self.pred_s1_WLM.forward(img_ORI)
            img_WLM = np.array(wholeLineMask1[0] * img_ORI[0]).reshape(img_ORI.shape)
            self.pred_s2_maskrcnn.forward(img_WLM, data_id)
    def pred_line(self):
        from matplotlib import pyplot as plt
        for data_id in range(98):
            print(data_id)
            img_ORI, img_name = self.load_img(self.dataset[data_id])
            print(img_name)
            wholeLineMask1 = self.pred_s1_WLM.forward(img_ORI)
            img_WLM = np.array(wholeLineMask1[0] * img_ORI[0]).reshape(img_ORI.shape)
            det_full_masks, det_bbox, bbox_center, det_id = self.pred_s2_maskrcnn.forward(img_WLM, data_id, False)
            line = fit_line(bbox_center)
            plt.figure()
            plt.imshow(img_ORI.squeeze())
            plt.plot(line[:, 0], line[:, 1], '.')
            plt.savefig("E:\CSI2019\AASCE_stage4_Process\data\selected2\plot_line/" + img_name)
    def pred_line_box(self):
        for data_id in range(98):
            img_ORI, filename = self.load_img(self.dataset[data_id])
            img_name = str(filename) + '.jpg'
            print(data_id)
            img_ORI, img_name = self.load_img(self.dataset[data_id])
            wholeLineMask1 = self.pred_s1_WLM.forward(img_ORI)
            img_WLM = np.array(wholeLineMask1[0] * img_ORI[0]).reshape(img_ORI.shape)
            det_full_masks, det_bbox, bbox_center, det_id = self.pred_s2_maskrcnn.forward(img_WLM, data_id, False)
            refine_box = self.pred_s3_RB.forward(img_ORI, det_bbox, bbox_center, det_full_masks)

            line = fit_line(bbox_center)
            line_refine_box = find_line_box(line, refine_box)
            np.save("E:\CSI2019\AASCE_stage4_Process\data\selected2\labels/" + img_name + ".npy", line_refine_box)

    def pred_refind_box(self):
        from matplotlib import pyplot as plt

        # data_id = 74
        # data_id = 77
        # data_id = 84
        for data_id in range(15, 98):
            print(data_id)
            img_ORI, filename = self.load_img(self.dataset[data_id])
            img_name = str(filename)
            print(img_name)
            wholeLineMask1 = self.pred_s1_WLM.forward(img_ORI)
            img_WLM = np.array(wholeLineMask1[0] * img_ORI[0]).reshape(img_ORI.shape)
            det_full_masks, det_bbox, bbox_center, det_id = self.pred_s2_maskrcnn.forward(img_WLM, data_id, False)
            boxMask1 = self.pred_s2_BM.forward(img_ORI)
            if not data_id in (74, 83, 77):
                bbox_center, correct_index = outlier_filter(bbox_center)
                det_bbox = det_bbox[correct_index]
            bm_label = mask2_label(boxMask1, wholeLineMask1)
            rough_label = label_distance_filter(bm_label, distance=(2000))
            label_2 = np.concatenate([bbox_center, rough_label])
            bbox_center = bbox_center[2:-1]
            det_bbox = det_bbox[2:-1]
            refine_box = self.pred_s3.forward(img_ORI, det_bbox, bbox_center, det_full_masks, boxMask1)

            # # bbox_center, _ = outlier_filter(bbox_center)

            # label_2 = label_2[label_2[:, 1].argsort()]
            # # label_2, _ = outlier_filter(label_2)
            line = fit_line_segment(label_2)
            # label_2 = line_outlier_filter(line, label_2)
            #
            line = fit_line_segment(label_2, deg1=8, extent=400, deg2a=1, deg2b=1, poly_num=5, stack=1)
            # final_label = bbox_center[2:-1]
            # bbox_center = label_interplot(bbox_center)
            # # line_refine_box = find_line_label_fix(line, 60)
            # line_refine_box = find_line_label(line, final_label)
            line_refine_box = find_line_box(line, refine_box)
            plt.figure(figsize=(4, 8))
            plt.imshow(img_ORI.squeeze())
            # plt.plot(line[:, 0], line[:, 1])
            for i, label in enumerate(np.concatenate(line_refine_box)):
                if i % 2 == 0:
                    plt.scatter(label[0], label[1], color='r', s=10)
                if i % 2 == 1:
                    plt.scatter(label[0], label[1], color='r', s=10)
            plt.savefig("E:\CSI2019\AASCE_stage4_Process\data\selected2\plot_line13/" + img_name)
            np.save("E:\CSI2019\AASCE_stage4_Process\data\selected2\labels/" + img_name + ".npy", line_refine_box)
            plt.close('all')


def label_distance(label4):
    x1 = label4[:,0].min()
    y1 = label4[:,1].min()
    x2 = label4[:,0].max()
    y2 = label4[:,1].max()
    distance = np.sqrt(((np.array([x1, y1])-np.array([x2,y2]))**2).sum())
    return distance

def resize_and_crop(img, dsize):
    img_new = np.zeros([len(img), dsize[0], dsize[1]])
    for i, im in enumerate(img):
        im = cv2.resize(im, (dsize[1], dsize[0]))
        im = np.array(im)
        img_new[i] = im
    return img_new
def resize_and_crop_pil(img, dsize):
    img_new = np.zeros([len(img), dsize[0], dsize[1]])
    for i, im in enumerate(img):
        im = Image.fromarray(im.astype('float32'))
        im = im.resize(np.array([dsize[1], dsize[0]]).astype(int), Image.ANTIALIAS)
        im = np.array(im)
        img_new[i] = im
    return img_new
def norm_index(line, y_label):
    lineY = line[:, 1]
    if y_label in lineY:
        indexY = np.where(lineY == y_label)[0][0]
        line1 = line[:, [1, 0]]
        norm_vector = line1[indexY+1] - line1[indexY-1]
        norm_vector = norm_vector / sum(norm_vector**2)
        return np.array(norm_vector*np.array([1,-1]))
    else:
        print('out of line')
def ylabel_index(line, y_label):
    lineY = line[:, 1]
    if y_label in lineY:
        indexY = np.where(lineY == y_label)[0][0]
        return line[indexY]
    else:
        print('out of line')
def fit_line(center_labels):
    left_x_origin = center_labels[:, 1].astype('int')
    left_y = center_labels[:, 0]
    extent = 200
    poly = np.polyfit(left_x_origin, left_y, deg=15)
    left_x_mid = np.linspace(left_x_origin[0]-extent, left_x_origin[-1]+extent, num=left_x_origin[-1] - left_x_origin[0]+extent*2+1)
    left_z_mid = np.polyval(poly, left_x_mid)

    return np.array([left_z_mid, left_x_mid]).transpose()

def fit_line_segment0(center_labels):
    left_x_origin = center_labels[:, 1].astype('int')
    left_y = center_labels[:, 0]
    poly = np.polyfit(left_x_origin, left_y, deg=8)
    left_x_mid = np.linspace(left_x_origin[0], left_x_origin[-1], num=left_x_origin[-1] - left_x_origin[0] + 1)
    left_z_mid = np.polyval(poly, left_x_mid)

    extent = 400
    z_begin = left_x_origin.min()-extent
    z_end = left_x_origin.max()+extent
    poly = np.polyfit(left_x_origin[0:2], left_y[0:2], deg=1)
    left_up_x = np.linspace(z_begin, left_x_origin[1], num=left_x_origin[1] - z_begin + 1)
    left_up_z = np.polyval(poly, left_up_x)

    poly = np.polyfit(left_x_origin[-2:], left_y[-2:], deg=1)
    left_down_x = np.linspace(left_x_origin[-2], z_end, num=z_end - left_x_origin[-2] + 1)
    left_down_z = np.polyval(poly, left_down_x)

    left_up_stack = int(left_x_origin[1] - left_x_origin[0] + 1)
    left_down_stack = int(left_x_origin[-1] - left_x_origin[-2] + 1)
    left_z_mid[0: left_up_stack] = left_up_z[0 - left_up_stack:]
    left_z_mid[0 - left_down_stack:] = left_down_z[0: left_down_stack]
    left_z = np.concatenate([left_up_z[0:0 - left_up_stack], left_z_mid, left_down_z[left_down_stack:]], axis=0)
    left_x = np.linspace(z_begin, z_end, num=z_end - z_begin + 1)

    # left_x = np.linspace(0, z_length, num=z_length - 0 + 1)
    return np.array([left_z, left_x]).transpose()
def fit_line_segment(center_labels, deg1=9, extent=300, deg2a=1, deg2b=2, poly_num=5, stack=2):
    left_x_origin = center_labels[:, 1].astype('int')
    left_y = center_labels[:, 0]
    poly = np.polyfit(left_x_origin, left_y, deg=deg1)
    left_x_mid = np.linspace(left_x_origin[0], left_x_origin[-1], num=left_x_origin[-1] - left_x_origin[0] + 1)
    left_z_mid = np.polyval(poly, left_x_mid)


    # poly_num = 5
    # stack = 2

    z_begin = left_x_origin.min()-extent
    z_end = left_x_origin.max()+extent
    poly = np.polyfit(left_x_origin[0:poly_num], left_y[0:poly_num], deg=deg2a)
    left_up_x = np.linspace(z_begin, left_x_origin[stack], num=left_x_origin[stack] - z_begin + 1)
    left_up_z = np.polyval(poly, left_up_x)

    poly = np.polyfit(left_x_origin[-poly_num:], left_y[-poly_num:], deg=deg2b)
    left_down_x = np.linspace(left_x_origin[-stack], z_end, num=z_end - left_x_origin[-stack] + 1)
    left_down_z = np.polyval(poly, left_down_x)

    left_up_stack = int(left_x_origin[stack] - left_x_origin[0] + 1)
    left_down_stack = int(left_x_origin[-1] - left_x_origin[-stack] + 1)


    mean1, sigma1 = 0, 1
    half_gaus = left_up_stack
    half_gaus = int(half_gaus)
    gaus_len = 2 * half_gaus
    half_zero = left_up_stack - half_gaus
    x1 = np.linspace(mean1 - 6 * sigma1, mean1 + 6 * sigma1, gaus_len)
    gaus = normal_distribution(x1, mean1, sigma1)
    left_gaus = gaus[0:half_gaus]
    left_gaus = np.concatenate([np.zeros(half_zero), left_gaus])
    left_gaus = left_gaus[0: left_up_stack]
    right_gaus = gaus[half_gaus:]
    right_gaus = np.concatenate([np.zeros(half_zero), right_gaus])
    right_gaus = right_gaus[0: left_up_stack]
    left_z_mid[0: left_up_stack] = left_up_z[0 - left_up_stack:]*(1-left_gaus) + left_z_mid[0: left_up_stack]*left_gaus
    # left_z_mid[0: left_up_stack] = left_up_z[0 - left_up_stack:]

    mean1, sigma1 = 0, 1
    half_gaus = left_down_stack
    half_gaus = int(half_gaus)
    gaus_len = 2 * half_gaus
    half_zero = left_down_stack - half_gaus
    x1 = np.linspace(mean1 - 6 * sigma1, mean1 + 6 * sigma1, gaus_len)
    gaus = normal_distribution(x1, mean1, sigma1)
    left_gaus = gaus[0:half_gaus]
    left_gaus = np.concatenate([np.zeros(half_zero), left_gaus])
    left_gaus = left_gaus[0: left_down_stack]
    right_gaus = gaus[half_gaus:]
    right_gaus = np.concatenate([ right_gaus, np.zeros(half_zero)])
    right_gaus = right_gaus[0: left_down_stack]
    left_z_mid[0 - left_down_stack:] = left_down_z[0: left_down_stack]*(1-right_gaus) + left_z_mid[0 - left_down_stack:]*right_gaus
    # left_z_mid[0 - left_down_stack:] = left_down_z[0: left_down_stack]

    left_z = np.concatenate([left_up_z[0:0 - left_up_stack], left_z_mid, left_down_z[left_down_stack:]], axis=0)
    left_x = np.linspace(z_begin, z_end, num=z_end - z_begin + 1)
    return np.array([left_z, left_x]).transpose()

def normal_distribution(x, mean, sigma):
    return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(sigma)

def line_length(line_label):
    x = line_label[:, 0]
    y = line_label[:, 1]
    area_list = [np.sqrt((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2) for i in range(1, len(x))]
    return area_list, sum(area_list)
def line_cut(line, small_length_list, segm_length):
    tmp_length = 0
    cut_index = []
    boxcount = 0
    for small_length, label in zip(small_length_list, line):
        tmp_length += small_length
        # print(tmp_length)
        if segm_length*boxcount-5*small_length <= tmp_length <= segm_length*boxcount+5*small_length:
            cut_index.append(label)
            boxcount += 1
    return np.array(cut_index)
def labels_order(label4):
    '''
    l0   l1
    l2   l3
    :param label4:
    :return:
    '''
    tmp1 = label4[label4[:,1].argsort()]
    tmp2 = label4[label4[:,0].argsort()]

    up = tmp1[[0,1]]
    down = tmp1[[2,3]]
    left = tmp2[[0,1]]
    right = tmp2[[2,3]]

    for label1 in label4:
        if label1 in up and label1 in left:
            l0 = label1
        elif label1 in up and label1 in right:
            l1 = label1
        elif label1 in down and label1 in right:
            l2 = label1
        else:
            l3 = label1
    # if tmp[0][0] <= tmp[1][0]:
    #     l0 = tmp[0]
    #     l1 = tmp[1]
    # else:
    #     l0 = tmp[1]
    #     l1 = tmp[0]
    # if tmp[2][0] <= tmp[3][0]:
    #     l2 = tmp[2]
    #     l3 = tmp[3]
    # else:
    #     l2 = tmp[3]
    #     l3 = tmp[2]
    label4 = np.array([l0, l1, l2, l3])
    return label4

def find_line_box(line, refine_box):
    line_refine_box = []
    for i, rbox in enumerate(refine_box):
        # print(i)
        # rbox = labels_order(rbox)

        l0l1 = rbox[0:2]
        l2l3 = rbox[2:4]
        center_l0l1 = l0l1.mean(0)
        center_l2l3 = l2l3.mean(0)

        Y_l0l1 = int(center_l0l1[1])
        Y_l2l3 = int(center_l2l3[1])

        # center_l0l1 = ylabel_index(line, Y_l0l1)
        # center_l2l3 = ylabel_index(line, Y_l2l3)

        Norm1_l0l1 = norm_index(line, Y_l0l1)
        Norm1_l2l3 = norm_index(line, Y_l2l3)
        # Norm2_l0l1 = 0-Norm1_l0l1
        # Norm2_l2l3 = 0-Norm1_l2l3
        # dis_l0l1 = label_distance(l0l1)
        dis_l0l1 = 300
        # dis_l2l3 = label_distance(l2l3)
        dis_l2l3 = 300
        line_l0 = center_l0l1 - Norm1_l0l1 * dis_l0l1
        line_l1 = center_l0l1 + Norm1_l0l1 * dis_l0l1
        line_l2 = center_l2l3 - Norm1_l2l3 * dis_l2l3
        line_l3 = center_l2l3 + Norm1_l2l3 * dis_l2l3

        line_refine_box.append(np.array([line_l0, line_l1, line_l2, line_l3]))
    return line_refine_box
# def find_line_label_box(line, labels, box):


def find_line_label(line, labels, label_extent=1):
    line_refine_box = []
    for i, label in enumerate(labels):
        # print(i)
        # if i in range(0,3) :
        #     continue
        Y_l0l1 = int(label[1]-label_extent)
        Y_l2l3 = int(label[1]+label_extent)
        center_l0l1 = ylabel_index(line, Y_l0l1)
        center_l2l3 = ylabel_index(line, Y_l2l3)
        Norm1_l0l1 = norm_index(line, Y_l0l1)
        Norm1_l2l3 = norm_index(line, Y_l2l3)

        dis_l0l1 = 300
        dis_l2l3 = 300
        line_l0 = center_l0l1 - Norm1_l0l1*dis_l0l1
        line_l1 = center_l0l1 + Norm1_l0l1*dis_l0l1
        line_l2 = center_l2l3 - Norm1_l2l3*dis_l2l3
        line_l3 = center_l2l3 + Norm1_l2l3*dis_l2l3

        line_refine_box.append(np.array([line_l0, line_l1, line_l2, line_l3]))
    return line_refine_box

def find_line_label_fix(line, box_num=40):
    line_refine_box = []
    box_num = box_num
    dedicate = int(box_num/20)
    small_length_list, length = line_length(line)
    segm_length = length/(box_num-1)
    cut_index = line_cut(line, small_length_list, segm_length)
    for i in range(box_num):
        if i in range(0, dedicate) or i in range(box_num-dedicate, box_num):
            continue
        print(i)
        cut_center = cut_index[i]


        Y_l0l1 = int(cut_center[1]-50)
        Y_l2l3 = int(cut_center[1]+50)
        center_l0l1 = ylabel_index(line, Y_l0l1)
        center_l2l3 = ylabel_index(line, Y_l2l3)
        Norm1_l0l1 = norm_index(line, Y_l0l1)
        Norm1_l2l3 = norm_index(line, Y_l2l3)

        dis_l0l1 = 300
        dis_l2l3 = 300
        line_l0 = center_l0l1 - Norm1_l0l1*dis_l0l1
        line_l1 = center_l0l1 + Norm1_l0l1*dis_l0l1
        line_l2 = center_l2l3 - Norm1_l2l3*dis_l2l3
        line_l3 = center_l2l3 + Norm1_l2l3*dis_l2l3

        line_refine_box.append(np.array([line_l0, line_l1, line_l2, line_l3]))
    return line_refine_box
# vis line
# plt.figure()
# plt.imshow(img_ORI.squeeze())
# plt.plot(line[:, 0], line[:, 1])
# for i, label in enumerate(np.concatenate(line_rb)):
#     if i % 2 == 0:
#         plt.scatter(label[0], label[1], color='r', s=10)
#     if i % 2 == 1:
#         plt.scatter(label[0], label[1], color='r', s=10)

def label_interplot(labels):
    last_len_labels = 0
    while len(labels) != last_len_labels:
        flag = 0
        last_len_labels = len(labels)
        labels = labels[labels[:, 1].argsort()]
        for i, label in enumerate(labels):
            if i == 0 or i == len(labels)-1 or flag == 1:
                continue
            dis = labels[i+1][1] - label[1]
            dis_last = label[1] - labels[i-1][1]
            if dis > 1.5*dis_last:
                label_new = np.array([label[0],label[1]+dis_last]).reshape([-1, 2])
                labels = np.concatenate([labels, label_new], axis=0)
                print('new', label_new)
                flag = 1
            # print(dis, dis_last)
    return labels

def even_size(size):
    if size[1] % 2:
        width = size[1] - 1
    else:
        width = size[1]
    if size[0] % 2:
        height = size[0] - 1
    else:
        height = size[0]
    return np.array([height, width])

class stage_pred_function():
    '''
    stage 1w, 1s, 2
    '''
    def __init__(self, net, data_size):
        self.net = net
        # self.net.eval()
        self.data_size = data_size

    def forward(self, img_input):
        img_input = np.array(img_input, copy=True)
        img_oriSize = img_input.squeeze().shape
        if img_input.ndim == 2:
            img_input = img_input.reshape((-1,)+img_input.shape)

        img_input = resize_and_crop(img_input, dsize=self.data_size)
        img_input = torch.from_numpy(img_input).cuda().type(torch.cuda.FloatTensor).unsqueeze(0)
        pred_output = self.net(img_input)

        pred_output1 = resize_and_crop(pred_output[0].cpu().detach().numpy(), dsize=img_oriSize)
        del pred_output
        return pred_output1


class pred_stage3_refineBox():
    '''
    stage3
    '''

    def __init__(self, net_box, net_point, data_size):
        self.net_box = net_box
        self.net_point = net_point
        self.data_size = data_size

    def forward(self, img_ORI, det_bbox, crop_center, det_full_masks, box_masks):
        label_f = []
        heigths = det_bbox[:, 3] - det_bbox[:, 1]
        widths = det_bbox[:, 2] - det_bbox[:, 0]
        for i in range(len(det_bbox)):
            crop_anchor = crop_center[i]
            crop_bbox = np.array([widths[i], heigths[i]]) + 150
            img_crop, begin_crop = img_region_crop(img_ORI, crop_anchor, crop_bbox)
            img_crop = resize_and_crop(img_crop, self.data_size)
            img_input = torch.from_numpy(img_crop).cuda().type(torch.cuda.FloatTensor).unsqueeze(0)
            boxpred_output = self.net_box(img_input)
            boxpred_mask = boxpred_output.cpu().detach().numpy()[0][0]
            pointpred_output = self.net_point(img_input)
            pointpred_mask = pointpred_output.cpu().detach().numpy()*boxpred_mask.squeeze()
            # full_mask_crop, begin_crop = img_region_crop(det_full_masks[i], crop_anchor, crop_bbox)
            # full_mask_crop = resize_and_crop(full_mask_crop, self.data_size)
            # pred_mask = (pred_mask/pred_mask.max() + full_mask_crop.squeeze())/2

            # pred_mask = (full_mask_crop.squeeze())
            # pred_mask = np.expand_dims(pred_mask, axis=2)

            pred_mask_label = self._mask2_label(pointpred_mask).squeeze()
            # pred_mask_label = min_rectangle(pred_mask)
            # pred_mask_label = approxPoly(pred_mask)

            height_scale = self.data_size[0] / crop_bbox[1]
            width_scale = self.data_size[1] / crop_bbox[0]

            pred_mask_label = pred_mask_label / np.array([width_scale, height_scale])
            dis = label_distance(pred_mask_label)
            if dis<55:
                print(i,'ooo')
            label_f.append(pred_mask_label+begin_crop)
            # plot_img_and_label(img_crop.squeeze(), min_rect)
        return label_f

    def _mask2_label(self, pred_mask):

        labels = masks_2_labels_thread(pred_mask)
        return labels

def min_rectangle(pred_mask):
    image = np.concatenate((pred_mask, pred_mask, pred_mask), axis=-1)
    image = np.array(image / image.max() * 255).astype('uint8')
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    last_dis = 0
    final_min_rect = np.array([])
    for i in range(length):
        cnt = contours[i]
        min_rect = cv2.minAreaRect(cnt)
        min_rect = np.array(cv2.boxPoints(min_rect))[[1,0,2,3]]
        # print(min_rect)
        new_dis = label_distance(min_rect)
        if new_dis > last_dis:
            final_min_rect = min_rect
            last_dis = new_dis
    return final_min_rect

def approxPoly(pred_mask):
    image = np.concatenate((pred_mask, pred_mask, pred_mask), axis=-1)
    image = np.array(image / image.max() * 255).astype('uint8')
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    approx_label = []
    for ie, ep in enumerate([0.008, 0.009,
                             0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
                             0.11, 0.12, 0.13, 0.14, 0.15]):
        for i in range(length):
            cnt = contours[i]
            # epsilon = 8
            epsilon = ep*cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            approx = np.array(approx).squeeze()
            if approx.shape[0] == 4:
                # print(epsilon)
                approx_label.append(approx)
            else:
                pass

    return np.array(approx_label).mean(0)

def mask2_label(boxMask, SLM):
    boxMask = np.expand_dims(boxMask, axis=0)
    boxMask_SLM = boxMask[0]*SLM[0]
    # bm_label = masks_2_labels_maximum(np.expand_dims(boxMask_SLM, axis=0))
    bm_label = masks_2_labels_thread(np.expand_dims(boxMask_SLM, axis=0))
    return bm_label.squeeze()

def box2_label(boxLabels):
    x1 = boxLabels[:, 0]
    y1 = boxLabels[:, 1]
    x2 = boxLabels[:, 2]
    y2 = boxLabels[:, 3]
    x = np.mean([x1, x2], axis=0)
    y = np.mean([y1, y2], axis=0)
    bbox_center = np.array([x, y]).transpose()
    order_list = bbox_center[:, 1].argsort()
    bbox_center = bbox_center[order_list]
    return bbox_center, order_list


# from sklearn.neighbors import LocalOutlierFactor
def outlier_filter(labels0):
    # clf = LocalOutlierFactor(n_neighbors=5, contamination=0.1)
    labels = np.array(labels0[labels0[:, 1].argsort()], copy=True)

    # y_pred = clf.fit_predict(labels)
    labels_len = 0
    std_mean0 = 0
    std_min0 = 0
    flag = 0
    error_point = np.zeros(len(labels))
    while len(labels) != labels_len:
        labels_len = len(labels)
        # labels = labels[labels[:, 1].argsort()]
        distance_mean = np.zeros(labels_len)
        distance_min = np.zeros(labels_len)
        for i, label in enumerate(labels):
            if i == 0:
                near_a = 1
                near_b = 2
            elif i == labels_len-1:
                near_b = labels_len-2
                near_a = near_b-1
            else:
                near_a = i-1
                near_b = i+1
            distance_mean[i] = np.sqrt(((labels[i] - labels) ** 2).sum(1)[[near_a, near_b]].mean())
            distance_min[i] = np.sqrt(((labels[i] - labels) ** 2).sum(1)[[near_a, near_b]].min())
            # print(distance[i])
        if flag == 0:
            std_mean0 = np.std(distance_mean)
            std_mean0 = 200 if 200 > std_mean0 else std_mean0
            std_min0 = np.std(distance_min)
        suspect_mean = distance_mean > std_mean0 * 3
        suspect_min = distance_min > np.std(distance_min) * 3
        suspect_ = suspect_mean * suspect_min
        print(np.where(suspect_ == 1))

        if suspect_.sum() > 0:
            if flag == 0:
                error_point += suspect_
            else:
                error_point += np.insert(suspect_, np.where(error_point > 0)[0], 0)
        labels = labels[np.array(1 - suspect_).astype(bool)]
        print(len(labels))
        flag += 1
    labels_f = labels0[np.array(1-error_point,dtype=bool)]
    # labels = labels[np.array(1 - suspect_).astype(bool)]
    return labels_f, np.array(1-error_point,dtype=bool)

def line_outlier_filter(line, labels, dis=75):
    error_idex = []
    labels = labels[labels[:, 1].argsort()]
    for i, label in enumerate(labels):
        min_dis = np.sqrt((((line - label) ** 2).sum(1))).min()
        if min_dis > dis:
            error_idex.append(i)
            print('delete', i)
    labels = np.delete(labels, error_idex, axis=0)
    return labels
def label_distance_filter(bm_label, distance=2000):
    bm_label = bm_label[bm_label[:, 1].argsort()]
    bm_dedian = np.median(bm_label[:, 1])

    bm_label = bm_label[(bm_label[:, 1] > bm_dedian-distance) & (bm_label[:, 1] < bm_dedian+distance)]
    # label_len_last = 0
    # while len(bm_label) != label_len_last & len(bm_label) >= 6:
    #     label_len_last = len(bm_label)
    # bm_label = correct_label_detect(bm_label, dif)

    return bm_label

# def correct_label_detect(bm_label, dif=500):
#     distance = bm_label[1:] - bm_label[0:-1]
#     distanceY = distance[:, 1]
#     errorY1 = np.array(distanceY > dif).astype(int)
#     errorY2 = np.array(distanceY < -dif).astype(int)
#     error_ind = np.convolve(errorY1 - errorY2, np.array([-1, 1]), mode='full')
#     correct_label = bm_label[abs(error_ind) != 2]
#     return correct_label

# def correct_label_detect(bm_label, dif=500):
#     distance = bm_label[1:] - bm_label[0:-1]
#     distanceY = distance[:, 1]
#     errorY1 = np.array(distanceY > dif).astype(int)
#     error_id = np.where(errorY1 > 0)[0]
#
#     error_id_m = np.mean(error_id)
#     upper = error_id[error_id < error_id_m]
#     bellow = error_id[error_id > error_id_m]
#     if len(error_id)>1:
#         error_id_bellow = bellow.max()+1
#         error_id_upper = upper.min()+1
#         bm_label = bm_label[error_id_bellow:error_id_upper]
#     elif len(error_id) == 1:
#         bm_center = len(bm_label)/2
#         error_id_m = error_id_m + 1
#         if error_id_m > bm_center:
#             bm_label = bm_label[:error_id_m]
#         elif error_id_m < bm_center:
#             bm_label = bm_label[error_id_m:]
#         else:
#             print('emmm')
#
#     else:
#         bm_label = bm_label
#
#     # right_ind = 1 - np.concatenate((errorY1))
#     # correct_label = bm_label[right_ind.astype(bool)]
#     return bm_label


def region_estimate(correct_label):
    standard_size = np.array((2300, 850))
    height_min = standard_size[0]
    width_min = standard_size[1]
    height_width_ratio = height_min/width_min
    labelX = correct_label[:, 0]
    labelX_left = labelX.min()
    labelX_right = labelX.max()
    label_width = labelX_right - labelX_left
    centerX = np.mean([labelX_left, labelX_right])
    labelY = correct_label[:, 1]
    labelY_up = labelY.min()
    labelY_down = labelY.max()
    label_height = labelY_down - labelY_up
    centerY = np.mean([labelY_up, labelY_down])

    anchor = np.array([centerX, centerY])

    exp_height = label_width * height_width_ratio
    exp_width = label_height / height_width_ratio
    if (label_width > width_min) & (label_height > height_min):

        if exp_height > label_height:
            label_height = exp_height
        elif exp_width > label_width:
            label_width = exp_width
        else:
            print('situation 1 error')

    elif (label_width > width_min) & (label_height <= height_min):
        label_height = exp_height
    elif (label_width <= width_min) & (label_height > height_min):
        label_width = exp_width
    elif (label_width <= width_min) & (label_height <= height_min):
        label_width = width_min
        label_height = height_min
    else:
        print('min size error')

    target_size = np.array([label_width, label_height])

    return anchor, target_size

def region_expand(target_size, exp=0.1):
    width_expand = 1 + exp
    height_expand = 1 + exp
    crop_size = target_size*(width_expand, height_expand)
    return crop_size

def Boundary_stripping(bm_label, img_shape, strip=300):
    standard_size = np.array((2300, 850))
    height_min = standard_size[0]
    width_min = standard_size[1]
    height_width_ratio = height_min/width_min

    bm_labelY = bm_label[:, 1]
    bm_labelX = bm_label[:, 0]
    img_height = img_shape[1]
    img_width = img_shape[2]
    up = bm_labelY.min()
    down = bm_labelY.max()
    left = bm_labelX.min()
    right = bm_labelX.max()
    up_d = up/img_height
    down_d = (img_height-down)/img_height

    region_up = strip*up_d
    if region_up > up:
        region_up = up
    region_down = img_height - strip*down_d
    if region_down < down:
        region_down = down

    # region_down = region_down + 300
    # region_left = strip*left_d
    # region_right = img_width - strip*right_d

    region_height = region_down - region_up
    region_width = region_height/height_width_ratio
    anchor = np.array([np.mean([left, right]), np.mean([region_up, region_down])]).astype(int)
    crop_size = np.array([region_width, region_height]).astype(int)

    return anchor, crop_size
    # return 0

def img_region_crop(img, anchor, crop_size):
    img = img.squeeze()
    img_shape = np.array(img.shape)
    margin = 1000
    crop_x = int(crop_size[0])
    # if crop_x%2:
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
    return img_crop, begin_crop

def confident(correct_label):
    exp = len(correct_label)/17
    return exp



if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # train_data_path = "E:\CSI2019\AASCE_stage2\OriginalData/train/"
    # test_data_path = "E:\CSI2019\AASCE_stage2\OriginalData/test/"
    # train_data_path = "E:\CSI2019\AASCE_stage3\DataScaled/train_test/"
    # train_data_path = "E:\CSI2019\AASCE_stage4_Process\data/0_DataOriginal/train_test/"
    # train_output_path = "E:\CSI2019\AASCE_stage4_Process\data_output/train_test/labels_mat/"
    # train_dataset = TrainORIDataset(data_path=train_data_path)
    # # train_dataset = CropDataset(data_path=test_data_path, crop_size=None, data_aug=False)
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # pred_data_path = "E:\CSI2019\AASCE_stage4_Process\data/0_DataOriginal\Archive/"
    # pred_output_path = "E:\CSI2019\AASCE_stage4_Process\data_output/Archive/labels_mat/"
    # pred_dataset = PredORIDataset(data_path=pred_data_path)
    # pred_loader = DataLoader(pred_dataset, batch_size=1, shuffle=False)

    # pred_data_path = "E:\CSI2019\AASCE_stage4_Process\data\selected2/"
    pred_data_path = "E:\CSI2019\AASCE_stage4_Process\data\ppt_selected2/"
    pred_output_path = "E:\CSI2019\AASCE_stage4_Process\data\selected2/labels_mat/"
    pred_dataset = PredORIDataset(data_path=pred_data_path)
    pred_loader = DataLoader(pred_dataset, batch_size=1, shuffle=False)

    dir_checkpoint = "E:\CSI2019\AASCE_stage4_Process/saved_files"
    log_dir = Path(dir_checkpoint+'logs')
    log_dir.mkdir(exist_ok=True, parents=True)
    logger = debug_logger(log_dir)
    logger.info(f'Device: {device}')





    # model = UNetWithResnet50Encoder(n_channels=1, n_classes=18).cuda()
    # model = NestedUNet(n_channels=1, n_classes=20)
    # model = SCSE_UNet(n_channels=1, n_classes=20)
    # model = SCSENestedUNet(n_channels=1, n_classes=18)
    # model = DilatedUNet(in_channels=1, classes=20)
    # model = ResUNet(in_channel=1, n_classes=5)

    # stage1_model_whole = ResUNet(in_channel=1, n_classes=2) #
    # root_dir = "E:\CSI2019\AASCE_stage4_Process/"
    # stage1_model_whole_load_dir = "saved_model/stage1_resunet_whole_line/CP200.pth"
    # stage1_model_whole.load_state_dict(torch.load(os.path.join(root_dir, stage1_model_whole_load_dir)))
    # logger.info('Stage1_Model loaded from {}'.format(stage1_model_whole_load_dir))

    root_dir = "E:\CSI2019\AASCE_stage4_Process/"

    stage1_model_whole = UNet(n_channels=1, n_classes=2) #
    stage1_model_whole_load_dir = "saved_model/stage1_unet_whole_line/CP324.pth"
    stage1_model_whole.load_state_dict(torch.load(os.path.join(root_dir, stage1_model_whole_load_dir)))
    logger.info('Stage1_Model loaded from {}'.format(stage1_model_whole_load_dir))


    stage1_model_segm = ResUNet(in_channel=1, n_classes=2) #
    stage1_model_segm_load_dir = "saved_model/stage1_resunet_segm_line/CP1127.pth"
    stage1_model_segm.load_state_dict(torch.load(os.path.join(root_dir, stage1_model_segm_load_dir)))
    logger.info('Stage1_Model loaded from {}'.format(stage1_model_segm_load_dir))

    stage2_model_box = UNet(n_channels=1, n_classes=18) #
    stage2_model_box_load_dir = "saved_model/stage2__unet_box/Bestmodel_568.pth"
    stage2_model_box.load_state_dict(torch.load(os.path.join(root_dir, stage2_model_box_load_dir)))
    logger.info('Stage2_Model loaded from {}'.format(stage2_model_box_load_dir))

    stage3_boxmodel = UNet(n_channels=1, n_classes=2) #
    stage3_model_load_dir = "saved_model/stage3_unet_refine_label/Bestmodel_485.pth"
    stage3_boxmodel.load_state_dict(torch.load(os.path.join(root_dir, stage3_model_load_dir)))
    logger.info('Stage3_Model loaded from {}'.format(stage3_model_load_dir))

    stage3_pointmodel = UNet(n_channels=1, n_classes=5) #
    stage3_model_load_dir = "saved_model\stage3_unet_refine_point_mask\Bestmodel_394.pth"
    stage3_pointmodel.load_state_dict(torch.load(os.path.join(root_dir, stage3_model_load_dir)))
    logger.info('Stage3_Model loaded from {}'.format(stage3_model_load_dir))

    # stage3_model = SCSE_UNet(n_channels=1, n_classes=2) #
    # stage3_model_load_dir = "saved_model/stage3_scseunet_refine_label/Bestmodel_82.pth"
    # stage3_model.load_state_dict(torch.load(os.path.join(root_dir, stage3_model_load_dir)))
    # logger.info('Stage3_Model loaded from {}'.format(stage3_model_load_dir))


    stage1_model_whole.cuda()
    stage1_model_segm.cuda()
    stage2_model_box.cuda()
    stage3_boxmodel.cuda()
    stage3_pointmodel.cuda()
    cudnn.benchmark = True # faster convolutions, but more memory


    # pred_a = pred(s1_modelw=stage1_model_whole,
    #       s1_models=stage1_model_segm,
    #       s2_model=stage2_model_box,
    #       stage3_model=stage3_model,
    #       dataLoader=train_loader,
    #       output_dir=train_output_path,
    #       )
    # pred_a.forward()
    pred_b = pred(s1_modelw=stage1_model_whole,
          s1_models=stage1_model_segm,
          s2_model=stage2_model_box,
          stage3_boxmodel=stage3_boxmodel,
          stage3_pointmodel=stage3_pointmodel,
          dataLoader=pred_loader,
          output_dir=pred_output_path,
          )
    pred_b.forward()

