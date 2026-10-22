"""Image/landmark pairing, geometric augmentation and paper-specific targets."""
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
import csv
import random
import re

import cv2
import numpy as np
from PIL import Image
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset

STAGES = {
    "wsm": (2, (690, 255)),
    "tsm": (2, (688, 254)),
    "boxes": (18, (806, 298)),
    "local-box": (2, (255, 255)),
    "local-points": (5, (255, 255)),
    "maskrcnn": (18, (806, 298)),
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path
    row: int | None = None  # normalized landmarks.csv row; None means pixel MAT/NPY


@lru_cache(maxsize=8)
def _csv_table(path, modified, size):
    table = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    if table.shape[1] != 136 or not np.isfinite(table).all():
        raise ValueError(f"{path}: expected finite N x 136 normalized x-then-y coordinates.")
    return table


def csv_table(path):
    stat = path.stat()
    return _csv_table(str(path), stat.st_mtime_ns, stat.st_size)


def discover(root, split=None, label_format="auto"):
    """Accept legacy split/images, native root/data/split, or data/split directly."""
    root = Path(root).resolve()
    if split is not None:
        image_dir, label_dir = root / "data" / split, root / "labels" / split
    elif root.parent.name == "data" and (root.parent.parent / "labels" / root.name).is_dir():
        image_dir, label_dir = root, root.parent.parent / "labels" / root.name
    else:
        image_dir, label_dir = root / "images", root / "labels"
    images = sorted(p for p in image_dir.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"No images found in {image_dir}; expected images/labels or data/<split>/labels/<split>.")
    if len({p.stem.casefold() for p in images}) != len(images):
        raise ValueError(f"Duplicate image stems in {image_dir}.")
    csv_rows = {}
    csv_path, names_path = label_dir / "landmarks.csv", label_dir / "filenames.csv"
    need_csv = label_format == "csv" or (label_format == "auto" and any(
        not any((label_dir / (name + ext)).is_file()
                for name in (image.name, image.stem) for ext in (".mat", ".npy")) for image in images))
    if need_csv and (csv_path.exists() or names_path.exists() or label_format == "csv"):
        if not (csv_path.exists() and names_path.exists()):
            raise ValueError(f"Both filenames.csv and landmarks.csv are required in {label_dir}.")
        table = csv_table(csv_path)
        with names_path.open(encoding="utf-8-sig", newline="") as stream:
            names = list(csv.reader(stream))
        if len(names) != len(table) or any(len(row) != 1 for row in names):
            raise ValueError(f"{names_path}: expected one image filename per landmark row.")
        for row, values in enumerate(names):
            name = values[0].strip().casefold()
            if name in csv_rows:
                raise ValueError(f"Duplicate filename in {names_path}: {name}")
            csv_rows[name] = row
    result = []
    for image in images:
        candidates = {label_dir / (name + ext)
                      for name in (image.name, image.stem) for ext in (".mat", ".npy")}
        labels = sorted(p for p in candidates if p.is_file())
        if label_format == "csv" or (label_format == "auto" and not labels and csv_rows):
            if image.name.casefold() not in csv_rows:
                raise ValueError(f"No CSV row for {image.name} in {names_path}.")
            result.append(Sample(image, csv_path, csv_rows[image.name.casefold()]))
        elif len(labels) != 1:
            raise ValueError(f"Expected exactly one .mat/.npy label for {image.name}, found {labels}.")
        else:
            result.append(Sample(image, labels[0]))
    return result


def case_id(sample):
    """Filename-derived acquisition/case group, not a verified patient identity."""
    prefix = r"sunhl-\d+th-\d{1,2}-[A-Za-z]{3}-\d{4}-"
    match = re.fullmatch(rf"({prefix}\d+)\s+[A-Z]\s+AP\d*(?:\s*\(\d+\))?", sample.image.stem, flags=re.IGNORECASE)
    if match is None:
        # These files do not expose a patient ID: conservatively keep all
        # coronal views from the same dated batch together.
        match = re.fullmatch(rf"({prefix}coronal)\d+", sample.image.stem, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Cannot infer a case group from {sample.image.name}; use explicit splits or --split-by image.")
    return match.group(1).casefold()


def split_training(samples, fraction=0.2, seed=2019, split_by="case"):
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between 0 and 1.")
    key = case_id if split_by == "case" else lambda sample: sample.image.stem.casefold()
    groups = sorted({key(sample) for sample in samples})
    if len(groups) < 2:
        raise ValueError("At least two independent groups are needed for a train/validation split.")
    random.Random(seed).shuffle(groups)
    n_val = min(len(groups) - 1, max(1, round(len(groups) * fraction)))
    selected = set(groups[:n_val])
    train = [s for s in samples if key(s) not in selected]
    val = [s for s in samples if key(s) in selected]
    return train, val


def check_split(train, val, group_cases=True):
    overlap = {s.image.stem.casefold() for s in train} & {s.image.stem.casefold() for s in val}
    if overlap:
        raise ValueError(f"Training/validation image IDs overlap: {sorted(overlap)[:5]}")
    if group_cases:
        known = []
        for samples in (train, val):
            groups = set()
            for sample in samples:
                try:
                    groups.add(case_id(sample))
                except ValueError:
                    continue
            known.append(groups)
        if known[0] & known[1]:
            raise ValueError("Related filename case groups overlap between training and validation.")


def load_points(sample, size, coordinate_base=0):
    w, h = size
    if sample.row is not None:
        values = csv_table(sample.label)[sample.row]
        points = np.column_stack((values[:68] * w, values[68:] * h))
    elif sample.label.suffix.lower() == ".mat":
        values = loadmat(sample.label)
        if "p2" not in values:
            raise ValueError(f"{sample.label}: missing MATLAB variable 'p2'.")
        points = values["p2"]
    else:
        points = np.load(sample.label, allow_pickle=False)
    points = np.asarray(points, dtype=np.float32)
    if points.shape == (2, 68):
        points = points.T
    if points.shape not in ((68, 2), (17, 4, 2)):
        raise ValueError(f"{sample.label}: expected (68,2), (2,68) or (17,4,2), got {points.shape}.")
    points = points.reshape(17, 4, 2).copy()
    if sample.row is None:
        points -= coordinate_base  # normalized CSV is scaled directly, never shifted
    if not np.isfinite(points).all() or (points < 0).any() or (points[..., 0] >= w).any() or (points[..., 1] >= h).any():
        raise ValueError(f"{sample.label}: coordinates must be finite pixel (x,y), within {w}x{h}.")
    if (np.diff(points.mean(axis=1)[:, 1]) <= 0).any():
        raise ValueError(f"{sample.label}: 17 vertebrae must be ordered superior to inferior.")
    for quad in points:
        if cv2.contourArea(quad[[0, 1, 3, 2]]) < 1:
            raise ValueError(f"{sample.label}: degenerate vertebral quadrilateral.")
    return points


def load_sample(sample, coordinate_base=0):
    with Image.open(sample.image) as im:
        points = load_points(sample, im.size, coordinate_base)
        image = np.asarray(im.convert("L"), dtype=np.float32)
    image /= max(float(image.max()), 1.0)  # legacy per-image maximum normalization
    return image, points


def audit_samples(samples, coordinate_base=0):
    """Preflight before workers start. Never change coordinates, order or source files."""
    valid, rejected = [], []
    for sample in samples:
        try:
            with Image.open(sample.image) as image:
                load_points(sample, image.size, coordinate_base)
                image.verify()
            valid.append(sample)
        except (ValueError, OSError) as error:
            rejected.append({"image": str(sample.image), "label": str(sample.label),
                             "label_row": sample.row, "reason": str(error)})
    return valid, {"total": len(samples), "accepted": len(valid), "rejected": rejected}


def polygons(points, size):
    masks = np.zeros((len(points), *size), dtype=np.float32)
    for mask, quad in zip(masks, points):
        cv2.fillPoly(mask, [np.rint(quad[[0, 1, 3, 2]]).astype(np.int32)], 1.0)
    return masks


def with_background(foreground):
    foreground = np.clip(foreground, 0, 1)
    # Overlapping quadrilaterals/Gaussians must still form a valid distribution.
    foreground = foreground / np.maximum(foreground.sum(axis=0, keepdims=True), 1.0)
    background = np.maximum(1.0 - foreground.sum(axis=0, keepdims=True), 0.0)
    return np.concatenate((foreground, background), axis=0).astype(np.float32)


def spine_mask(points, size, margin, blur):
    """Reconstructed continuous envelope. Margin/blur are fractions of vertebral width."""
    h, w = size
    # Connect all left/right endplates to include intervertebral gaps.
    left = points[:, [0, 2], :].reshape(-1, 2)
    right = points[:, [1, 3], :].reshape(-1, 2)
    left, right = left[np.argsort(left[:, 1])], right[np.argsort(right[:, 1])]
    contour = np.concatenate((left, right[::-1]))
    mask = np.zeros((h, w), np.float32)
    cv2.fillPoly(mask, [np.rint(contour).astype(np.int32)], 1.0)
    width = max(float(np.median(np.linalg.norm(points[:, 1] - points[:, 0], axis=1))), 1.0)
    radius = min(max(h, w), max(0, round(width * margin)))
    if radius:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        mask = cv2.dilate(mask, kernel)
    sigma = width * blur
    if sigma > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_CONSTANT)
    return mask[None]


def point_heatmaps(points, size, sigma):
    yy, xx = np.mgrid[:size[0], :size[1]]
    maps = []
    for x, y in points:
        dist2 = (xx - x) ** 2 + (yy - y) ** 2
        maps.append(np.where(dist2 <= (3 * sigma) ** 2, np.exp(-dist2 / (2 * sigma ** 2)), 0))
    return np.asarray(maps, np.float32)


def transform(image, points, size, augment=False, local_index=None, crop_margin=150, jitter=0.1):
    h, w = image.shape
    out_h, out_w = size
    if local_index is None:
        matrix = np.array([[out_w / w, 0, 0], [0, out_h / h, 0]], np.float32)
        selected = points.copy()
    else:
        selected = points[local_index:local_index + 1].copy()
        lo, hi = selected.min(axis=(0, 1)), selected.max(axis=(0, 1))
        extent = hi - lo + crop_margin
        center = (lo + hi) / 2
        if augment:
            center += np.array([random.uniform(-jitter, jitter), random.uniform(-jitter, jitter)]) * extent
        start = center - extent / 2
        matrix = np.array([[out_w / extent[0], 0, -start[0] * out_w / extent[0]],
                           [0, out_h / extent[1], -start[1] * out_h / extent[1]]], np.float32)
    if augment:
        affine = cv2.getRotationMatrix2D((out_w / 2, out_h / 2), random.uniform(-5, 5), random.uniform(0.9, 1.1))
        matrix = (np.vstack((affine, [0, 0, 1])) @ np.vstack((matrix, [0, 0, 1])))[:2].astype(np.float32)
    result = cv2.warpAffine(image, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    selected = selected @ matrix[:, :2].T + matrix[:, 2]
    if augment and random.random() < 0.5:
        result = result[:, ::-1].copy()
        selected[..., 0] = out_w - 1 - selected[..., 0]
        selected = selected[:, [1, 0, 3, 2]]  # preserve TL,TR,BL,BR channel semantics
    return result.astype(np.float32), selected.astype(np.float32), matrix


class SpineDataset(Dataset):
    def __init__(self, samples, stage, size=None, augment=False, coordinate_base=0,
                 crop_margin=150, jitter=0.1, sigma=6.67, wsm_margin=0.75,
                 wsm_blur=0.20, tsm_margin=0.25, tsm_blur=0.08):
        self.samples, self.stage = samples, stage
        self.size = tuple(size or STAGES[stage][1])
        self.augment, self.coordinate_base = augment, coordinate_base
        self.crop_margin, self.jitter, self.sigma = crop_margin, jitter, sigma
        self.mask_params = (wsm_margin, wsm_blur) if stage == "wsm" else (tsm_margin, tsm_blur)

    def __len__(self):
        return len(self.samples) * (17 if self.stage.startswith("local-") else 1)

    def __getitem__(self, index):
        local = self.stage.startswith("local-")
        sample = self.samples[index // 17 if local else index]
        image, points = load_sample(sample, self.coordinate_base)
        image, points, _ = transform(image, points, self.size, self.augment,
                                    index % 17 if local else None, self.crop_margin, self.jitter)
        if self.stage == "maskrcnn":
            masks = polygons(points, self.size).astype(np.uint8)
            boxes, keep = [], []
            for i, mask in enumerate(masks):
                ys, xs = np.nonzero(mask)
                if len(xs) and xs.max() > xs.min() and ys.max() > ys.min():
                    keep.append(i)
                    boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            if not keep:
                raise ValueError(f"No visible vertebrae in {sample.image} after transform.")
            target = {"boxes": torch.tensor(boxes, dtype=torch.float32),
                      "labels": torch.tensor(np.array(keep) + 1, dtype=torch.int64),
                      "masks": torch.from_numpy(masks[keep]),
                      "image_id": torch.tensor(index, dtype=torch.int64),
                      "area": torch.from_numpy(masks[keep].sum(axis=(1, 2)).astype(np.float32)),
                      "iscrowd": torch.zeros(len(keep), dtype=torch.int64)}
            return torch.from_numpy(image[None].repeat(3, axis=0)), target
        if self.stage in ("wsm", "tsm"):
            foreground = spine_mask(points, self.size, *self.mask_params)
        elif self.stage == "local-points":
            foreground = point_heatmaps(points[0], self.size, self.sigma)
        else:
            foreground = polygons(points, self.size)
        return torch.from_numpy(image[None]), torch.from_numpy(with_background(foreground))


def detection_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def seed_worker(_):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
