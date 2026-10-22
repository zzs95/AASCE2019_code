"""Architectures and safe checkpoint loading (no MXNet dependency)."""
from collections.abc import Mapping
import torch
from torch import nn
from torch.nn import functional as F
from model import UNet, ResUNet
from .data import STAGES


def build_model(stage, arch="unet", size=None, pretrained=False, proposal_limit=None):
    if stage != "maskrcnn":
        return UNet(1, STAGES[stage][0]) if arch == "unet" else ResUNet(in_channel=1, n_classes=STAGES[stage][0])
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    h, w = size or STAGES[stage][1]
    kwargs = dict(min_size=min(h, w), max_size=max(h, w), box_detections_per_img=100)
    if proposal_limit:
        kwargs.update(rpn_pre_nms_top_n_train=proposal_limit * 2, rpn_post_nms_top_n_train=proposal_limit,
                      rpn_pre_nms_top_n_test=proposal_limit * 2, rpn_post_nms_top_n_test=proposal_limit,
                      box_batch_size_per_image=proposal_limit)
    model = maskrcnn_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None,
        weights_backbone=None, **kwargs)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 18)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, 18)
    return model


def load_weights(model, path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    weights = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(weights, Mapping):
        raise ValueError(f"Not a PyTorch state dict: {path}")
    model.load_state_dict(weights, strict=True)
    return checkpoint


class HeatmapLoss(nn.Module):
    """BCE + positive-region focal loss adapted from dental/refine_net."""
    def forward(self, pred, target):
        pred = pred.clamp(1e-6, 1 - 1e-6)
        bce = F.binary_cross_entropy(pred, target)
        selected = target > 0.01
        element_bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-element_bce)
        focal = 0.25 * (1 - pt).square() * element_bce
        positive = focal[selected].mean() if selected.any() else pred.sum() * 0
        return (bce + positive) / 2


class WholeSpineGate(nn.Module):
    """Frozen A prediction masks detector inputs, during BOTH training and validation."""
    def __init__(self, checkpoint, size=(690, 255)):
        super().__init__()
        self.net = UNet(1, 2)
        state = load_weights(self.net, checkpoint)
        config = state.get("config", {})
        if config and (config.get("stage") != "wsm" or config.get("arch") != "unet"):
            raise ValueError("The WSM gate requires U-Net A weights (--stage wsm --arch unet).")
        if config and tuple(config["image_size"]) != tuple(size):
            raise ValueError("--wsm-size must match the A checkpoint's training image size.")
        self.net.requires_grad_(False)
        self.size = size
        self.eval()

    @torch.no_grad()
    def forward(self, images):
        self.net.eval()
        result = []
        for image in images:
            small = F.interpolate(image[:1][None], self.size, mode="bilinear", align_corners=False)
            mask = self.net(small)[:, :1]
            mask = F.interpolate(mask, image.shape[-2:], mode="bilinear", align_corners=False)[0]
            result.append(image * mask)
        return result
