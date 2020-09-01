import random
import numpy as np
import torch

def masks_2_labels_maximum(masks):
    flag = torch_or_numpy(masks)
    batch_num = masks.shape[0]
    channel_num = masks.shape[1]-1
    if flag == 'torch':
        labels = torch.zeros(batch_num, channel_num, 2)
    else:
        labels = np.zeros((batch_num, channel_num, 2))
    for i in range(batch_num):
        for j in range(channel_num):
            if flag == 'torch':
                d = torch.eq(masks[i][j], masks[i][j].max())
                max_indx = torch.nonzero(d)[0]
            else:
                max_indx = np.array(np.where(masks[i][j] == masks[i][j].max()))[:, 0]
            max_indx = max_indx.reshape(2)
            labels[i,j,1] = max_indx[0]
            labels[i,j,0] = max_indx[1]
    return labels

def masks_2_labels_coordinate_weight(masks):
    if masks.dtype == torch.float32:
        masks = masks.type(torch.FloatTensor).numpy()
    batch_num = masks.shape[0]
    channel_num = masks.shape[1]-1
    masks_width = masks.shape[2]
    labels = np.zeros((batch_num, channel_num, 2))
    m = np.array(range(1, masks_width+1)).reshape(1, masks_width)
    m_x = np.repeat(m, masks_width, axis=0)
    # m_x = torch.from_numpy(m_x).type(torch.FloatTensor)
    m_y = np.repeat(m.transpose((1,0)), masks_width, axis=1)
    # m_y = torch.from_numpy(m_y).type(torch.FloatTensor)

    for i in range(batch_num):
        for j in range(channel_num):
            mask_thread_ = masks[i][j] > 0.5
            mx_masks = (m_x*masks[i][j])
            tm_x = mask_thread_ * mx_masks
            labels[i,j,1] = tm_x.sum()/(masks_width**2 - np.where(tm_x==0)[0].shape[0])
            my_masks = (m_y*masks[i][j])
            tm_y = mask_thread_ * my_masks
            labels[i,j,0] = tm_y.sum()/(masks_width**2 - np.where(tm_y==0)[0].shape[0])
    return labels

def masks_2_labels_thread(masks):
    thread = 0.5
    flag = torch_or_numpy(masks)
    batch_num = masks.shape[0]
    channel_num = masks.shape[1]-1
    if flag == 'torch':
        labels = torch.zeros(batch_num, channel_num, 2)
    else:
        labels = np.zeros((batch_num, channel_num, 2))
    for i in range(batch_num):
        for j in range(channel_num):
            mask = masks[i][j]
            mask = mask/mask.max()
            mask = mask > thread

            if flag == 'torch':
                mask_labels = torch.nonzero(mask).float().mean(dim=0)
                labels[i,j,0] = mask_labels[1]
                labels[i,j,1] = mask_labels[0]
            else:
                mask_labels = np.where(mask == mask.max())
                labels[i,j,0] = mask_labels[1].mean()
                labels[i,j,1] = mask_labels[0].mean()
    return labels

def masks_2_labels_max_tread(masks):
    max_label = masks_2_labels_maximum(masks)
    nor_thread_laebl = masks_2_labels_thread(masks)
    # nor_thread_laebl2 = masks_2_labels_normal_thread2(masks)
    mean_label = (max_label + nor_thread_laebl)/2
    return mean_label

def torch_or_numpy(masks):
    if torch.is_tensor(masks):
        # print('torch tensor')
        flag = 'torch'
    else:
        # print('numpy array')
        flag = 'numpy'
    return flag
