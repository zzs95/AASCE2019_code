"""Synthetic regression tests; no patient data or external model downloads."""
import gc
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
from scipy.io import savemat
import torch
from torch.utils.data import DataLoader

import train
from training.data import SpineDataset, discover, check_split, load_sample, transform, with_background
from training.engine import run_epoch, detection_iou
from training.models import build_model, HeatmapLoss, WholeSpineGate


def make_fixture(root):
    for split, names in (("train", ["train_0", "train_1"]), ("val", ["val_0"])):
        folder = Path(root) / split
        (folder / "images").mkdir(parents=True)
        (folder / "labels").mkdir()
        for i, name in enumerate(names):
            i += 3 if split == "val" else 0
            points = np.array([[[25 + i, 8 + j * 6], [53 + i, 8 + j * 6],
                                [25 + i, 12 + j * 6], [53 + i, 12 + j * 6]] for j in range(17)], np.float32)
            yy, xx = np.mgrid[:128, :80]
            image = (20 + 140 * np.exp(-((xx - 39) / 22) ** 2) + yy * 0.25).astype(np.uint8)
            for quad in points:
                cv2.fillPoly(image, [quad[[0, 1, 3, 2]].astype(np.int32)], 230)
            Image.fromarray(image).save(folder / "images" / (name + ".png"))
            if name.endswith("0"):
                savemat(folder / "labels" / (name + ".png.mat"), {"p2": points.reshape(68, 2)})
            else:
                np.save(folder / "labels" / (name + ".npy"), points.reshape(68, 2).T)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        make_fixture(cls.root)
        cls.samples = discover(cls.root / "train")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_pairing_shapes_coordinate_base_and_split(self):
        for sample in self.samples:
            image, points = load_sample(sample)
            self.assertEqual(points.shape, (17, 4, 2))
            self.assertEqual(image.shape, (128, 80))
        with self.assertRaisesRegex(ValueError, "overlap"):
            check_split(self.samples, self.samples)
        # Explicit MATLAB one-based input yields exactly the original pixel coordinates.
        sample = self.samples[0]
        _, reference = load_sample(sample)
        source = sample.label.read_bytes()
        try:
            savemat(sample.label, {"p2": reference.reshape(68, 2) + 1})
            _, shifted = load_sample(sample, 1)
            np.testing.assert_array_equal(reference, shifted)
        finally:
            sample.label.write_bytes(source)

    def test_targets_normalize_and_local_crop_geometry(self):
        for stage, channels in (("wsm", 2), ("tsm", 2), ("boxes", 18), ("local-box", 2), ("local-points", 5)):
            ds = SpineDataset(self.samples, stage, (96, 64), crop_margin=20, sigma=2)
            image, target = ds[0]
            self.assertEqual(tuple(target.shape), (channels, 96, 64))
            self.assertEqual(tuple(image.shape), (1, 96, 64))
            torch.testing.assert_close(target.sum(0), torch.ones(96, 64))
            self.assertGreater(float(target[0].max()), 0)
            self.assertTrue(torch.isfinite(target).all())
        wsm = SpineDataset(self.samples, "wsm", (96, 64))[0][1][0].sum()
        tsm = SpineDataset(self.samples, "tsm", (96, 64))[0][1][0].sum()
        self.assertGreater(float(wsm), float(tsm))
        image, points = load_sample(self.samples[0])
        _, cropped, matrix = transform(image, points, (96, 64), local_index=0, crop_margin=20)
        expected = points[:1] @ matrix[:, :2].T + matrix[:, 2]
        np.testing.assert_allclose(cropped, expected)
        maps = SpineDataset(self.samples, "local-points", (96, 64), crop_margin=20, sigma=2)[0][1]
        for channel, (x, y) in zip(maps[:4], cropped[0]):
            cy, cx = np.unravel_index(int(channel.argmax()), channel.shape)
            self.assertLessEqual(abs(cx - x), 1)
            self.assertLessEqual(abs(cy - y), 1)
        # Test horizontal flip swaps the anatomical corner channels consistently.
        with patch("training.data.random.uniform", side_effect=[0, 1]), patch("training.data.random.random", return_value=0):
            _, flipped, _ = transform(image, points, (128, 80), augment=True)
        expected = points[:, [1, 0, 3, 2]].copy()
        expected[..., 0] = 79 - expected[..., 0]
        np.testing.assert_allclose(flipped, expected)
        background = with_background(np.zeros((4, 8, 8), np.float32))
        self.assertTrue((background[-1] == 1).all())

    def test_five_unet_heads_backward_and_validation(self):
        for stage in ("wsm", "tsm", "boxes", "local-box", "local-points"):
            ds = SpineDataset(self.samples, stage, (64, 64), sigma=2)
            loader = DataLoader(ds, batch_size=2)
            model = build_model(stage)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            initial = next(model.parameters()).detach().clone()
            loss = run_epoch(model, loader, "cpu", False, optimizer, max_steps=1)
            self.assertTrue(np.isfinite(loss))
            self.assertFalse(torch.equal(initial, next(model.parameters())))
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            run_epoch(model, loader, "cpu", False, max_steps=1)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, state[key]), key)
            del model, optimizer, state
            gc.collect()

    def test_loss_handles_empty_positive_region(self):
        prediction = torch.full((1, 2, 4, 4), 0.5, requires_grad=True)
        loss = HeatmapLoss()(prediction, torch.zeros_like(prediction))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_detector_targets_and_real_resnet_backward(self):
        ds = SpineDataset(self.samples, "maskrcnn", (96, 64))
        image, target = ds[0]
        self.assertEqual(tuple(image.shape), (3, 96, 64))
        self.assertEqual(target["labels"].tolist(), list(range(1, 18)))
        self.assertEqual(target["masks"].dtype, torch.uint8)
        self.assertTrue(((target["boxes"][:, 2:] - target["boxes"][:, :2]) > 0).all())
        # No pretrained weights/network access; genuine ResNet50/FPN + RPN/ROI/mask heads.
        torch.manual_seed(2019)
        model = build_model("maskrcnn", size=(96, 64), proposal_limit=16)
        initial = model.roi_heads.mask_predictor.mask_fcn_logits.weight.detach().clone()
        del model
        gc.collect()
        wsm_path = self.root / "untrained_wsm_for_smoke.pth"
        torch.save(build_model("wsm").state_dict(), wsm_path)
        output = self.root / "detector"
        record = train.main(["--stage", "maskrcnn", "--train-dir", str(self.root / "train"),
                             "--val-dir", str(self.root / "val"), "--output-dir", str(output),
                             "--image-size", "96", "64", "--epochs", "1", "--batch-size", "1",
                             "--device", "cpu", "--workers", "1", "--threads", "2", "--no-augment",
                             "--max-steps", "1", "--max-val-steps", "1", "--proposal-limit", "16",
                             "--wsm-checkpoint", str(wsm_path), "--wsm-size", "64", "64"])
        self.assertTrue(np.isfinite(record["train_loss"]))
        self.assertTrue(0 <= record["val_mask_iou"] <= 1)
        weights = torch.load(output / "best_weights.pth", weights_only=True)
        self.assertFalse(torch.equal(initial, weights["roi_heads.mask_predictor.mask_fcn_logits.weight"]))
        del weights
        from predict_detector import TorchvisionSpineDetector
        predictor = TorchvisionSpineDetector(output / "best.pth", wsm_checkpoint=wsm_path, wsm_size=(64, 64), score_threshold=0)
        masks, boxes, centers, labels = predictor.forward(load_sample(self.samples[0])[0])
        self.assertEqual(tuple(masks.shape[1:]), (128, 80))
        self.assertEqual(tuple(boxes.shape), (len(masks), 4))
        self.assertEqual(tuple(centers.shape), (len(masks), 2))
        self.assertTrue(((labels >= 0) & (labels < 17)).all())
        self.assertTrue((boxes[:, [0, 2]] <= 80).all() and (boxes[:, [1, 3]] <= 128).all())
        predictor.score_threshold = 2
        empty = predictor.forward(load_sample(self.samples[0])[0])
        self.assertEqual(empty[0].shape, (0, 128, 80))
        perfect = {"labels": target["labels"], "masks": target["masks"][:, None].float(), "scores": torch.ones(17)}
        total, count = detection_iou([perfect], [target])
        self.assertEqual(total, count)
        del predictor
        gc.collect()

    def test_legacy_default_sizes_and_optional_resunet(self):
        model = build_model("wsm").eval()
        with torch.no_grad():
            for size in ((690, 255), (688, 254), (806, 298), (255, 255)):
                prediction = model(torch.zeros(1, 1, *size))
                self.assertEqual(tuple(prediction.shape), (1, 2, *size))
        del model
        model = build_model("tsm", "resunet").eval()
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.zeros(1, 1, 64, 48)).shape), (1, 2, 64, 48))
        del model
        gc.collect()

    def test_checkpoint_resume_equals_uninterrupted_and_wsm_gate(self):
        base = ["--stage", "wsm", "--train-dir", str(self.root / "train"), "--val-dir", str(self.root / "val"),
                "--device", "cpu", "--image-size", "64", "64", "--max-steps", "1", "--max-val-steps", "1", "--threads", "2"]
        continuous, resumed = self.root / "continuous", self.root / "resumed"
        train.main(base + ["--epochs", "2", "--output-dir", str(continuous)])
        train.main(base + ["--epochs", "1", "--output-dir", str(resumed)])
        train.main(base + ["--epochs", "2", "--output-dir", str(resumed), "--resume", str(resumed / "last.pth")])
        a = torch.load(continuous / "last.pth", weights_only=True)
        b = torch.load(resumed / "last.pth", weights_only=True)
        self.assertEqual(a["epoch"], 1)
        self.assertEqual(a["scheduler"], b["scheduler"])
        for key in a["model"]:
            self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)
        self.assertEqual(a["best"], b["best"])
        history = [json.loads(s) for s in (resumed / "history.jsonl").read_text().splitlines()]
        self.assertEqual([s["epoch"] for s in history], [1, 2])
        gate = WholeSpineGate(resumed / "best_weights.pth", (64, 64))
        gated = gate([torch.ones(3, 96, 64)])[0]
        self.assertEqual(tuple(gated.shape), (3, 96, 64))
        self.assertFalse(gated.requires_grad)
        self.assertTrue((gated >= 0).all() and (gated <= 1).all())
        with self.assertRaisesRegex(ValueError, "already contains"):
            train.main(base + ["--epochs", "3", "--output-dir", str(resumed)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
