import torch
from torch.autograd import Function, Variable
import torch.nn as nn
import torch.nn.functional as F

class IoU_comput(Function):
    def forward(self, input):
        self.save_for_backward(input)
        eps = 0.0001
        stack_value = 0
        for i, mask in enumerate(input):
            eval_mask = torch.FLoarTensor(torch.abs(mask) > 0.0001)
            back_mask = torch.FLoarTensor(input[0:i, i+1:].sum(0) > 0.0001)
            stack_value += torch.FLoarTensor(eval_mask * back_mask).sum()
        height, width = input[-2:]
        iou_value = (stack_value + eps)/(2 * height * width)
        return iou_value

def iou_loss(input):
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input)):
        s = s + IoU_comput().forward(c)
    return s / (i + 1)