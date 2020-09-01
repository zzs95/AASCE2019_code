import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, reduce=True, ignore=0.01):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.ignore = ignore

    def forward(self, preds, labels):
        # mask = labels != self.ignore
        mask = labels > self.ignore
        labels = labels[mask]
        preds = preds[mask]
        if self.logits:
            bce_loss = F.binary_cross_entropy_with_logits(preds, labels, reduction='none')
        else:
            bce_loss = F.binary_cross_entropy(preds, labels, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss

        if self.reduce:
            return torch.mean(focal_loss)
        else:
            return focal_loss

def focal_loss(preds, labels):
    s = FocalLoss().forward(preds, labels)
    return s