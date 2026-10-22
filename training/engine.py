"""Epoch training, held-out validation and resumable, weights-only-safe checkpoints."""
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from .models import HeatmapLoss, WholeSpineGate, load_weights


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def rng_state(generators):
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loaders": [g.get_state() for g in generators]}


def restore_rng(state, generators):
    random.setstate(state["python"])
    name, array, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(array, np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
    for generator, value in zip(generators, state["loaders"]):
        generator.set_state(value.cpu())


@torch.no_grad()
def detection_iou(predictions, targets):
    """Mean GT instance mask IoU, matched by vertebral tag; missing tags score zero.

    Diagnostic validation metric, NOT COCO AP or the paper's Cobb-angle SMAPE.
    """
    total, count = 0.0, 0
    for prediction, target in zip(predictions, targets):
        for label, truth in zip(target["labels"], target["masks"]):
            candidates = torch.where((prediction["labels"] == label) & (prediction["scores"] >= 0.05))[0]
            count += 1
            if candidates.numel():
                index = candidates[prediction["scores"][candidates].argmax()]
                mask = prediction["masks"][index, 0] >= 0.5
                truth = truth.bool()
                intersection = (mask & truth).sum().item()
                union = (mask | truth).sum().item()
                total += intersection / max(union, 1)
    return total, count


def run_epoch(model, loader, device, detection, optimizer=None, gate=None,
              max_steps=None, grad_clip=5.0):
    training = optimizer is not None
    model.train(training)
    criterion = HeatmapLoss()
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for step, (images, targets) in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break
            if detection:
                images = [im.to(device) for im in images]
                targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
                if gate is not None:
                    images = gate(images)
            else:
                images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            if detection and not training:
                value, n = detection_iou(model(images), targets)
                total += value
                count += n
                continue
            if detection:
                parts = model(images, targets)
                loss = sum(parts.values())
            else:
                loss = criterion(model(images), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at batch {step + 1}: {loss.item()}")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
                optimizer.step()
            n = len(images)
            total += float(loss.detach()) * n
            count += n
    if count == 0:
        raise ValueError("Empty epoch: no samples were processed.")
    mean = total / count
    return 1 - mean if detection and not training else mean


def fit(args, model, train_loader, val_loader, generators, manifest):
    device = torch.device(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((output / name).exists() for name in ("last.pth", "best.pth", "history.jsonl")):
        raise ValueError(f"{output} already contains a run. Use --resume or a new --output-dir.")
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=args.lr_patience)
    detection = args.stage == "maskrcnn"
    gate = WholeSpineGate(args.wsm_checkpoint, tuple(args.wsm_size)).to(device) if args.wsm_checkpoint else None
    start, best, bad_epochs = 0, math.inf, 0
    config = vars(args).copy()
    if args.init_weights:
        load_weights(model, args.init_weights, device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError("--resume requires a full training checkpoint; use --init-weights for a raw state dict.")
        for key in ("stage", "arch", "image_size", "coordinate_base", "crop_margin", "crop_jitter", "sigma",
                    "wsm_margin", "wsm_blur", "tsm_margin", "tsm_blur", "wsm_checkpoint", "wsm_size",
                    "augment", "batch_size", "workers", "seed", "proposal_limit", "lr_patience", "early_stopping",
                    "lr", "weight_decay", "grad_clip", "max_steps", "max_val_steps", "threads"):
            if checkpoint["config"][key] != config[key]:
                raise ValueError(f"Resume configuration mismatch for {key}: {checkpoint['config'][key]} vs {config[key]}")
        for key in ("label_format", "invalid_samples", "data_root", "train_split", "val_split", "val_fraction", "split_by"):
            if checkpoint["config"].get(key) != config[key]:
                raise ValueError(f"Resume data configuration mismatch for {key}; use --init-weights to start a new data pipeline.")
        if checkpoint["manifest"] != manifest:
            raise ValueError("Resume data split changed; start a new run with --init-weights instead.")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Move optimizer buffers explicitly after loading a portable CPU checkpoint.
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        scheduler.load_state_dict(checkpoint["scheduler"])
        start, best = checkpoint["epoch"] + 1, checkpoint["best"]
        bad_epochs = checkpoint["bad_epochs"]
        restore_rng(checkpoint["rng"], generators)
        history = output / "history.jsonl"
        if history.exists() and history.read_text(encoding="utf-8").strip():
            last_record = json.loads(history.read_text(encoding="utf-8").strip().splitlines()[-1])
            if last_record["epoch"] >= start + 1:
                raise ValueError("Output history is newer than the resumed checkpoint; use a new output directory.")
    if start >= args.epochs:
        raise ValueError(f"Checkpoint already completed {start} epochs; --epochs is the TOTAL requested epoch count.")
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "split.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for epoch in range(start, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        train_loss = run_epoch(model, train_loader, device, detection, optimizer, gate,
                               args.max_steps, args.grad_clip)
        val_objective = run_epoch(model, val_loader, device, detection, gate=gate, max_steps=args.max_val_steps)
        scheduler.step(val_objective)
        improved = val_objective < best
        best, bad_epochs = (val_objective, 0) if improved else (best, bad_epochs + 1)
        record = {"epoch": epoch + 1, "lr": lr, "train_loss": train_loss,
                  "val_objective": val_objective, "best_objective": best}
        if detection:
            record["val_mask_iou"] = 1 - val_objective
        state = {"format_version": 1, "epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "best": best, "bad_epochs": bad_epochs, "config": config,
                 "manifest": manifest, "rng": rng_state(generators)}
        atomic_save(state, output / "last.pth")
        if improved:
            atomic_save(state, output / "best.pth")
            atomic_save(model.state_dict(), output / "best_weights.pth")
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if args.early_stopping and bad_epochs >= args.early_stopping:
            print(f"Early stopping at epoch {epoch + 1}.", flush=True)
            break
    return record
