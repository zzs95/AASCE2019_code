import torch.nn as nn
import torch.nn.functional as F
import torch


class BCELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        # probs = torch.sigmoid(logits)  # 二分类问题，sigmoid等价于softmax
        probs = logits
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)