import numpy as np
import mxnet as mx

import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv.data import batchify

from gluoncv.data.transforms import bbox as tbbox
from gluoncv.data.transforms import image as timage
from gluoncv.data.transforms import mask as tmask


class stage2_maskrcnn():
    def __init__(self):
        # training contexts
        self.ctx = mx.gpu(0)

        # network
        net_name = '_'.join(('mask_rcnn', 'resnet50_v1b', 'coco'))
        pretrained = 'E:\CSI2019\AASCE_stage4_Process\saved_model\stage2_mask_rcnn\mask_rcnn_resnet50_v1b_coco_0551_0.0000.params'
        self.net = gcv.model_zoo.get_model(net_name, pretrained=False)
        self.net.load_parameters(pretrained.strip())
        self.net.collect_params().reset_ctx(self.ctx)
        self.clipper = gcv.nn.bbox.BBoxClipToImage()
        self.rcnntransform = rcnn_transform(self.net.short, self.net.max_size)


    def forward(self, img_ORI0, img_id, viz_result_flag=False):
        from main import outlier_filter, box2_label, label_distance_filter
        _, ORI_height , ORI_width = img_ORI0.shape
        img_ORI = mx.ndarray.array(img_ORI0)
        img_ORI = img_ORI / img_ORI.max()
        x_scaled, img_info = self.rcnntransform(img_ORI)
        x_scaled = x_scaled.expand_dims(0)
        x_scaled = x_scaled.as_in_context(self.ctx)
        ids, scores, bboxes, masks = self.net(x_scaled)
        det_bbox = self.clipper(bboxes, x_scaled)

        det_bbox = det_bbox[0].asnumpy()
        det_id = ids[0].asnumpy()
        det_score = scores[0].asnumpy()
        det_mask = masks[0].asnumpy()
        im_height, im_width, im_scale = img_info.asnumpy()
        valid = np.where(((det_id >= 0) & (det_score >= 0)))[0]
        det_id = det_id[valid]
        det_score = det_score[valid]
        det_bbox = det_bbox[valid] / im_scale
        det_mask = det_mask[valid]

        # fill full mask
        im_height, im_width = int(ORI_height), int(ORI_width)
        det_full_masks = []
        for bbox, mask in zip(det_bbox, det_mask):
            det_full_masks.append(gdata.transforms.mask.fill(mask, bbox, (im_width, im_height)))
        det_full_masks = np.array(det_full_masks)

        # iou squeeze
        bbox_num = len(det_bbox)
        iou_matrix = np.zeros([bbox_num, bbox_num])
        for ib, bbox_current in enumerate(det_bbox):
            for ib_2 in range(ib + 1, bbox_num):
                bbox_test = det_bbox[ib_2]
                iou_score = iou(bbox_current, bbox_test)
                iou_matrix[ib, ib_2] = iou_score
        iou_matrix1 = iou_matrix >= 0.3
        stack_id = np.array(np.where(iou_matrix1 > 0)).transpose()
        for s_id in stack_id:
            det_full_masks[s_id[0]] += det_full_masks[s_id[1]]
            det_full_masks[s_id[0]] = np.array(det_full_masks[s_id[0]] > 0).astype(np.uint8)
            if det_score[s_id[0]] < det_score[s_id[1]]:
                det_score[s_id[0]] = det_score[s_id[1]]
        det_full_masks = np.delete(det_full_masks, stack_id[:, 1], axis=0)
        det_bbox = np.delete(det_bbox, stack_id[:, 1], axis=0)
        det_score = np.delete(det_score, stack_id[:, 1], axis=0)
        det_id = np.delete(det_id, stack_id[:, 1], axis=0)

        # stack region
        bbox_num = len(det_bbox)
        iou_matrix = np.zeros([bbox_num, bbox_num])
        for ib, bbox_current in enumerate(det_bbox):
            for ib_2 in range(ib + 1, bbox_num):
                bbox_test = det_bbox[ib_2]
                iou_score = iou(bbox_current, bbox_test)
                iou_matrix[ib, ib_2] = iou_score
        iou_matrix2 = iou_matrix > 0.0001
        stack_id = np.array(np.where(iou_matrix2 > 0)).transpose()
        for s_id in stack_id:
            if det_full_masks[s_id[0]].sum() >= det_full_masks[s_id[1]].sum():
                det_full_masks[s_id[0]] = det_full_masks[s_id[0]] - det_full_masks[s_id[1]]
                det_full_masks[s_id[0]] = np.array(det_full_masks[s_id[0]] > 0).astype(np.uint8)
            else:
                det_full_masks[s_id[1]] = det_full_masks[s_id[1]] - det_full_masks[s_id[0]]
                det_full_masks[s_id[1]] = np.array(det_full_masks[s_id[1]] > 0).astype(np.uint8)

        # outlier filter
        bbox_center, order_list = box2_label(det_bbox)
        out_filter = 0
        if out_filter:
            bbox_center, keep_index = outlier_filter(bbox_center)
            det_bbox = det_bbox[order_list][keep_index]
            det_full_masks = det_full_masks[order_list][keep_index]
            det_id = det_id[order_list][keep_index]
        else:
            det_bbox = det_bbox[order_list]
            det_full_masks = det_full_masks[order_list]
            det_id = det_id[order_list]


        if viz_result_flag:
            viz_result(img_ORI, det_full_masks, det_bbox, det_id, img_id)
        else:
            return det_full_masks, det_bbox, bbox_center, det_id

def viz_result(img_ORI, det_full_masks, det_bbox, det_id, img_id):
    CLASSES = ['1j', '2j', '3j', '4j', '5j', '6j', '7j', '8j', '9j', '10j', '11j', '12j', '13j', '14j', '15j', '16j', '17j']
    train_image = img_ORI.transpose([1,2,0])
    from matplotlib import pyplot as plt
    from gluoncv.utils import viz
    plt.ioff()
    width, height = train_image.shape[1], train_image.shape[0]
    train_masks = det_full_masks
    plt_image = viz.plot_mask(train_image*255, train_masks)
    fig = plt.figure(figsize=(5, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax = viz.plot_bbox(plt_image, det_bbox, labels=det_id, class_names=CLASSES, ax=ax)
    # plt.show()
    fig.savefig(str(img_id)+'.jpg')

def iou(box1, box2):
    '''
    两个框（二维）的 iou 计算
    注意：边框以左上为原点
    box:[top, left, bottom, right]
    box:[l,t,r,b]
    '''

    in_h = min(box1[3], box2[3]) - max(box1[1], box2[1])
    in_w = min(box1[2], box2[2]) - max(box1[0], box2[0])
    inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
    union = (box1[3] - box1[1]) * (box1[2] - box1[0]) + \
            (box2[3] - box2[1]) * (box2[2] - box2[0]) - inter
    iou = inter / union
    return iou

class rcnn_transform(object):
    def __init__(self, short=600, max_size=1000,
                 net=None, multi_stage=False, **kwargs):
        self._max_size = max_size
        self._short = short

        self._anchors = None
        self._multi_stage = multi_stage
        if net is None:
            return

    def __call__(self, src):
        src = src.transpose([1,2,0])
        h, w, _ = src.shape
        img_resize = timage.resize_short_within(src, self._short, self._max_size, interp=1)
        im_scale = float(img_resize.shape[0]) / h

        imgs = [img_resize.squeeze(), img_resize.squeeze(), img_resize.squeeze()]
        img_resize = mx.nd.stack(*imgs, axis=0)
        img_resize = mx.nd.image.normalize(img_resize, mean=0.5, std=0.22)
        img_info = mx.nd.array([img_resize.shape[-2], img_resize.shape[-1], im_scale])

        return img_resize, img_info

if __name__ == '__main__':
    stage2_maskrcnn()