"""Train one of the six independently optimized models in the AASCE paper."""
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from training.data import (STAGES, SpineDataset, discover, check_split, detection_collate,
                           seed_worker, audit_samples, split_training, case_id)
from training.engine import fit
from training.models import build_model


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage", required=True, choices=STAGES)
    p.add_argument("--data-root", help="Native boostnet_labeldata root (data/training, labels/training).")
    p.add_argument("--train-split", default="training", help="Training split under --data-root.")
    p.add_argument("--val-split", help="Optional existing validation split; default: split only training data.")
    p.add_argument("--val-fraction", type=float, default=0.2, help="Fraction of training groups reserved for validation.")
    p.add_argument("--split-by", choices=("case", "image"), default="case", help="Filename case groups for automatic splitting.")
    p.add_argument("--train-dir", help="Explicit split/images + split/labels, or native data/training directory.")
    p.add_argument("--val-dir", help="Explicit independent validation directory, paired as above.")
    p.add_argument("--label-format", choices=("auto", "mat", "csv"), default="auto", help="Auto prefers pixel MAT/NPY, then normalized CSV.")
    p.add_argument("--invalid-samples", choices=("error", "skip"), default="error", help="Preflight policy; every rejected sample is recorded.")
    p.add_argument("--check-data", action="store_true", help="Write data audit/split only, without constructing or training models.")
    p.add_argument("--output-dir", help="Defaults to saved_model/<stage>.")
    p.add_argument("--arch", choices=("unet", "resunet"), default="unet", help="Heatmap networks only. Legacy B uses resunet.")
    p.add_argument("--image-size", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"), help="Defaults to legacy stage input size.")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    p.add_argument("--epochs", type=int, default=200, help="Total epoch count, including resumed epochs.")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--lr-patience", type=int, default=10)
    p.add_argument("--early-stopping", type=int, default=40, help="Zero disables early stopping.")
    p.add_argument("--workers", type=int, default=0, help="Windows-safe default; increase after checking RAM.")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=2019)
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--coordinate-base", type=int, choices=(0, 1), default=0, help="Set 1 for MATLAB one-based pixel coordinates.")
    p.add_argument("--crop-margin", type=float, default=150, help="Added TOTAL width/height in source pixels, as in legacy inference.")
    p.add_argument("--crop-jitter", type=float, default=0.1, help="Training-only center jitter as a fraction of crop extent.")
    p.add_argument("--sigma", type=float, default=6.67, help="Gaussian sigma in output heatmap pixels.")
    p.add_argument("--wsm-margin", type=float, default=0.75, help="Envelope dilation radius / median vertebral width.")
    p.add_argument("--wsm-blur", type=float, default=0.20, help="Gaussian sigma / median vertebral width.")
    p.add_argument("--tsm-margin", type=float, default=0.25)
    p.add_argument("--tsm-blur", type=float, default=0.08)
    p.add_argument("--wsm-checkpoint", help="Frozen U-Net A weights used to mask detector input.")
    p.add_argument("--wsm-size", nargs=2, type=int, default=[690, 255], metavar=("HEIGHT", "WIDTH"))
    p.add_argument("--allow-unmasked-detector", action="store_true", help="Explicit ablation/smoke-test deviation from paper.")
    p.add_argument("--pretrained-detector", action="store_true", help="Download COCO initialization for Mask R-CNN.")
    p.add_argument("--proposal-limit", type=int, help="Reduce detector proposal counts for smoke tests; omit for training.")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--resume", help="Full checkpoint, including optimizer, scheduler and RNG state.")
    group.add_argument("--init-weights", help="Strict model weights initialization; starts a new optimization run.")
    p.add_argument("--max-steps", type=int, help="Debug: limit training batches per epoch.")
    p.add_argument("--max-val-steps", type=int, help="Debug: limit validation batches; omit for real model selection.")
    return p


def manifest(samples):
    result = []
    hashes = {}
    def digest(path):
        if path not in hashes:
            hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        return hashes[path]
    for sample in samples:
        result.append({"image": str(sample.image), "label": str(sample.label),
                       "label_row": sample.row,
                       "image_sha256": digest(sample.image), "label_sha256": digest(sample.label)})
    return result


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    a.image_size = a.image_size or list(STAGES[a.stage][1])
    a.output_dir = str(Path(a.output_dir or Path("saved_model") / a.stage).resolve())
    for key in ("data_root", "train_dir", "val_dir", "resume", "init_weights", "wsm_checkpoint"):
        if getattr(a, key):
            setattr(a, key, str(Path(getattr(a, key)).resolve()))
    for key in ("epochs", "batch_size", "threads", "lr", "grad_clip", "sigma"):
        if getattr(a, key) <= 0:
            p.error(f"--{key.replace('_', '-')} must be positive")
    for key in ("workers", "weight_decay", "crop_margin", "crop_jitter", "wsm_margin", "wsm_blur", "tsm_margin", "tsm_blur", "early_stopping", "lr_patience"):
        if getattr(a, key) < 0:
            p.error(f"--{key.replace('_', '-')} cannot be negative")
    for key in ("max_steps", "max_val_steps", "proposal_limit"):
        if getattr(a, key) is not None and getattr(a, key) <= 0:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if min(*a.image_size, *a.wsm_size) < 32:
        p.error("Image sizes must be at least 32 pixels in each dimension")
    if a.data_root:
        if a.train_dir or a.val_dir:
            p.error("Use --data-root OR --train-dir/--val-dir, not both")
        if not 0 < a.val_fraction < 1:
            p.error("--val-fraction must be between zero and one")
        for split in (a.train_split, a.val_split):
            if split and (Path(split).name != split or split in (".", "..")):
                p.error("Split names must be single directory names")
        root = Path(a.data_root)
        if not (root / "data").is_dir() and (root / "boostnet_labeldata" / "data").is_dir():
            root = root / "boostnet_labeldata"
        a.data_root = str(root)
    elif not (a.train_dir and a.val_dir):
        p.error("Provide --data-root, or both --train-dir and --val-dir")
    elif a.val_split:
        p.error("--val-split requires --data-root")
    if a.stage == "maskrcnn" and a.arch != "unet":
        p.error("--arch applies only to heatmap models")
    if a.stage == "maskrcnn" and not a.check_data and not (a.wsm_checkpoint or a.allow_unmasked_detector):
        p.error("Mask R-CNN requires --wsm-checkpoint from stage A; ablation: --allow-unmasked-detector")
    if a.stage != "maskrcnn" and (a.wsm_checkpoint or a.pretrained_detector or a.proposal_limit or a.allow_unmasked_detector):
        p.error("Detector flags apply only to --stage maskrcnn")
    if a.pretrained_detector and (a.resume or a.init_weights):
        p.error("Do not combine COCO initialization with existing weights")
    a.device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    output = Path(a.output_dir)
    if not a.resume and any((output / name).exists() for name in ("last.pth", "best.pth", "history.jsonl")):
        raise ValueError(f"{output} already contains a run. Use --resume or a new --output-dir.")
    if a.data_root:
        source_train = discover(a.data_root, a.train_split, a.label_format)
    else:
        source_train = discover(a.train_dir, label_format=a.label_format)
    train_samples, train_audit = audit_samples(source_train, a.coordinate_base)
    audit = {"source_training": train_audit, "policy": a.invalid_samples,
             "coordinate_base": a.coordinate_base, "label_format": a.label_format,
             "reorders_or_clips_annotations": False}
    if not train_samples:
        output.mkdir(parents=True, exist_ok=True)
        (output / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        raise ValueError("No valid 68-point training samples; inspect data_audit.json. Unsupported point counts are never converted.")
    if a.data_root and not a.val_split:
        train_samples, val_samples = split_training(train_samples, a.val_fraction, a.seed, a.split_by)
        audit["validation_source"] = "held out from training; original data/test is not used"
        audit["split_by"] = a.split_by
        if a.split_by == "case":
            audit["train_case_groups"] = len({case_id(s) for s in train_samples})
            audit["val_case_groups"] = len({case_id(s) for s in val_samples})
    else:
        source_val = (discover(a.data_root, a.val_split, a.label_format) if a.data_root
                      else discover(a.val_dir, label_format=a.label_format))
        val_samples, val_audit = audit_samples(source_val, a.coordinate_base)
        audit["source_validation"] = val_audit
        audit["validation_source"] = "explicit independent split"
    output.mkdir(parents=True, exist_ok=True)
    audit["train_images"], audit["validation_images"] = len(train_samples), len(val_samples)
    (output / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    rejected = sum(len(audit[key]["rejected"]) for key in ("source_training", "source_validation") if key in audit)
    print(f"Data preflight: training={len(train_samples)}, validation={len(val_samples)}, rejected={rejected}; audit={output / 'data_audit.json'}", flush=True)
    if rejected and a.invalid_samples == "error":
        raise ValueError(f"{rejected} invalid samples; inspect data_audit.json, then use --invalid-samples skip to explicitly exclude them.")
    if not train_samples or not val_samples:
        raise ValueError("Training and validation must both contain valid samples.")
    check_split(train_samples, val_samples, group_cases=a.split_by == "case")
    split_manifest = {"data_pipeline_version": 2, "train": manifest(train_samples), "val": manifest(val_samples)}
    if {s["image_sha256"] for s in split_manifest["train"]} & {s["image_sha256"] for s in split_manifest["val"]}:
        p.error("Identical image files occur in both training and validation splits")
    if a.wsm_checkpoint:
        split_manifest["wsm_sha256"] = hashlib.sha256(Path(a.wsm_checkpoint).read_bytes()).hexdigest()
    if a.check_data:
        (output / "split.json").write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Data check complete; no model was created or trained.", flush=True)
        return audit
    common = dict(stage=a.stage, size=a.image_size, coordinate_base=a.coordinate_base,
                  crop_margin=a.crop_margin, jitter=a.crop_jitter, sigma=a.sigma,
                  wsm_margin=a.wsm_margin, wsm_blur=a.wsm_blur, tsm_margin=a.tsm_margin, tsm_blur=a.tsm_blur)
    train = SpineDataset(train_samples, augment=a.augment, **common)
    val = SpineDataset(val_samples, augment=False, **common)
    generators = [torch.Generator().manual_seed(a.seed), torch.Generator().manual_seed(a.seed + 1)]
    kwargs = dict(batch_size=a.batch_size, num_workers=a.workers, worker_init_fn=seed_worker,
                  pin_memory=a.device.startswith("cuda"), collate_fn=detection_collate if a.stage == "maskrcnn" else None)
    train_loader = DataLoader(train, shuffle=True, generator=generators[0], **kwargs)
    val_loader = DataLoader(val, shuffle=False, generator=generators[1], **kwargs)
    print(f"Stage={a.stage}, device={a.device}, train={len(train)}, validation={len(val)}, HxW={a.image_size}", flush=True)
    model = build_model(a.stage, a.arch, a.image_size, a.pretrained_detector, a.proposal_limit)
    return fit(a, model, train_loader, val_loader, generators, split_manifest)


if __name__ == "__main__":
    main()
