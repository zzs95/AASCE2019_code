"""Adapter for NEW torchvision detector weights; the legacy MXNet script is unchanged."""
import numpy as np
import torch
from torch.nn import functional as F
from training.models import WholeSpineGate, build_model, load_weights


class TorchvisionSpineDetector:
    def __init__(self, checkpoint, device="cpu", wsm_checkpoint=None, size=None,
                 wsm_size=(690, 255), score_threshold=0.5):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = state.get("config", {})
        if config.get("stage", "maskrcnn") != "maskrcnn":
            raise ValueError("Expected a Mask R-CNN checkpoint.")
        size = size or config.get("image_size", (806, 298))
        self.size = tuple(size)
        self.device = torch.device(device)
        self.net = build_model("maskrcnn", size=size, proposal_limit=config.get("proposal_limit"))
        load_weights(self.net, checkpoint)
        self.net.to(self.device).eval()
        self.gate = WholeSpineGate(wsm_checkpoint, wsm_size).to(self.device) if wsm_checkpoint else None
        if config.get("wsm_checkpoint") and self.gate is None:
            raise ValueError("This detector was trained with WSM masking; supply wsm_checkpoint.")
        self.score_threshold = score_threshold

    @torch.no_grad()
    def forward(self, image, *, already_masked=False):
        """Return full masks, xyxy boxes, xy centers, zero-based vertebral tags.

        Input: grayscale HxW or 1xHxW array. Coordinates are returned in its
        original resolution. Set already_masked=True for the existing pipeline,
        which multiplies the WSM before invoking its detector.
        """
        array = np.asarray(image, np.float32).squeeze()
        if array.ndim != 2 or not np.isfinite(array).all() or (array < 0).any():
            raise ValueError("Expected a finite, nonnegative grayscale HxW image.")
        if array.max() > 1:
            array = array / array.max()
        tensor = torch.from_numpy(array.copy())[None].repeat(3, 1, 1).to(self.device)
        tensor = F.interpolate(tensor[None], self.size, mode="bilinear", align_corners=False)[0]
        images = self.gate([tensor]) if self.gate is not None and not already_masked else [tensor]
        result = self.net(images)[0]
        keep = []
        for label in range(1, 18):
            candidates = torch.where((result["labels"] == label) & (result["scores"] >= self.score_threshold))[0]
            if candidates.numel():
                keep.append(candidates[result["scores"][candidates].argmax()].item())
        boxes = result["boxes"][keep].cpu().numpy().reshape(-1, 4)
        boxes *= np.array([array.shape[1] / self.size[1], array.shape[0] / self.size[0]] * 2)
        if keep:
            maps = F.interpolate(result["masks"][keep], array.shape, mode="bilinear", align_corners=False)
            masks = (maps[:, 0] >= 0.5).cpu().numpy().astype(np.uint8)
        else:
            masks = np.zeros((0, *array.shape), np.uint8)
        labels = result["labels"][keep].cpu().numpy() - 1
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        order = np.argsort(centers[:, 1])
        return masks[order], boxes[order], centers[order], labels[order]
