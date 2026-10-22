# Training guide

This guide describes the PyTorch training pipeline. The historical MXNet/GluonCV
inference code remains available, with separate dependencies.

## Environment

Install the matched `torch==2.6.0` / `torchvision==0.21.0` pair and
`requirements-training.txt` as shown in the README. The code has been tested on
Windows with Python 3.12 and CPU wheels. For GPU training, install the matching
CUDA wheels first. Run commands from the repository root.

## Native dataset layout

```text
boostnet_labeldata/
  data/training/image.jpg
  data/test/test_image.jpg
  labels/training/image.jpg.mat
  labels/training/filenames.csv
  labels/training/landmarks.csv
  labels/training/angles.csv
  labels/test/...
```

Use `--data-root /path/to/boostnet_labeldata`; its parent directory is also
accepted when it contains a `boostnet_labeldata` subdirectory. No dataset files
are copied or renamed.

Pixel labels are a MATLAB variable **`p2`** or a numeric NumPy array, with shape
`(68,2)`, `(2,68)` or `(17,4,2)`. Coordinates are `(x,y)` in source image pixels.
Vertebrae must be ordered superior to inferior, and the four corners of each
vertebra are **top-left, top-right, bottom-left, bottom-right**. Default pixel
indexing is zero-based, matching the legacy loader; use `--coordinate-base 1`
for genuinely one-based MATLAB coordinates. Object NumPy arrays are rejected.

`--label-format auto` prefers MAT/NPY labels and falls back to CSV where they are
missing. `--label-format csv` explicitly selects:

- `filenames.csv`: one image filename per row.
- `landmarks.csv`: 136 columns per row, first 68 normalized x coordinates, then
  68 normalized y coordinates, paired by **filename**, not directory order.

CSV coordinates are multiplied by each source image's width/height and are not
shifted by `--coordinate-base`. `angles.csv` is not a landmark training target.

Existing `<split>/images` + `<split>/labels` datasets are still supported through
`--train-dir` and `--val-dir`. Those flags also accept native `data/<split>`
directories; the corresponding `labels/<split>` directory is found automatically.
Do not combine explicit split-directory flags with `--data-root`.

## Validation split and annotation audit

By default, `--data-root` uses the original **training split only**. It holds out
20% of its filename case groups with seed 2019 before generating local patches.
The original `data/test` is left out of training and model selection.

Related A/B/... AP views, lowercase names and `(2)` variants share a group.
Coronal images from the same dated batch are conservatively kept together.
These groups are inferred from filenames, not verified patient identities.
If true patient identifiers are available, create independent splits using them
and provide those splits explicitly. For other naming schemes, an explicit
`--split-by image` is available; it does not guarantee related-view separation.
An existing `data/validation` / `labels/validation` split can be selected with
`--val-split validation`.

All samples are checked before model construction or data workers start:

```powershell
$dataRoot = "D:/datasets/boostnet_labeldata"
python train.py --stage wsm --data-root "$dataRoot" --invalid-samples skip --check-data --output-dir saved_model/data-check
```

The default invalid-sample policy is **`error`**. Use **`--invalid-samples skip`**
to explicitly exclude invalid samples and retain their paths and rejection
reasons in `data_audit.json`. The code does not automatically reorder vertebrae,
clip coordinates or truncate point arrays. Unsupported historical 76/84/56-point
challenge annotations are rejected by this 68-point pipeline.

On the inspected local dataset, 457 of 481 training images passed the checks;
23 had nonmonotonic vertebral order and one had an out-of-bounds point. With
default grouping/seed, the usable data split into **374 training images (183
groups)** and **83 validation images (46 groups)**. These counts describe that
dataset copy and can change after annotation review. The original 128-image test
split had four order-check failures; it was inspected for format only.

## Train all stages

The README contains the six commands. Heatmap models A/B/C/D/E are optimized
independently. Train A before Mask R-CNN and pass its weights through
`--wsm-checkpoint`. The frozen A prediction masks detector inputs in both
training and validation; validation annotations do not generate input masks.
`--pretrained-detector` explicitly opts into downloading COCO initialization.
Without it, the detector starts from random weights and makes no download.

Local D/E crops are generated from ground-truth vertebral bounding boxes with
150 source pixels added to the total width and height, as in legacy inference.
Center jitter applies only during training. Patch targets use the same image and
coordinate transform, including the left/right corner swap for horizontal flips.

The default architectures are all U-Net. Legacy inference instantiates a
**ResUNet for B**: use `--arch resunet` for B to match it, or change its B instance
to `UNet(1,2)` before loading new default U-Net weights. All checkpoint paths in
the old inference script must be updated for the local installation.

## Parameter provenance

| Setting | Default / provenance |
| --- | --- |
| Output channels | Paper: A/B/D = 2, C = 18, E = 5; background last |
| Input height × width | Legacy `main.py`; overridable with `--image-size` |
| WSM/TSM targets | Reconstructed continuous spine envelope, dilation and blur |
| WSM radius / blur sigma | 0.75 / 0.20 × median vertebral width; configurable reconstruction |
| TSM radius / blur sigma | 0.25 / 0.08 × median vertebral width; configurable reconstruction |
| Point Gaussian sigma | 6.67 output pixels, truncated at 3 sigma; legacy 40-pixel kernel reference |
| Heatmap loss | `(BCE + Focal) / 2`, alpha 0.25, gamma 2, target threshold 0.01; adapted from dental training |
| SGD | LR 0.01, momentum 0.9, weight decay 1e-4; adapted from dental training |
| Epochs / batch size | 200 / 2; implementation defaults, not recovered paper settings |
| Training augmentation | Rotation ±5°, scale 0.9–1.1, horizontal flip, local center jitter ±10% |
| Gradient clipping | Norm 5 |
| LR schedule / early stopping | Halve after 10 unimproved validation epochs; stop after 40 |

The paper does not provide exact numerical WSM/TSM construction or a full
optimization recipe. These configurable choices are not claimed as the original
experiment settings. `--no-augment` disables training geometry and crop jitter;
validation always uses deterministic geometry.

## Checkpoints and resume

Each run saves:

- `last.pth`: last full checkpoint for resume.
- `best.pth`: best full checkpoint, including architecture/data configuration.
- `best_weights.pth`: raw model state dict for legacy heatmap loading.
- `config.json`, `split.json`, `data_audit.json`, `history.jsonl`.

```powershell
python train.py --stage wsm --data-root "$dataRoot" --invalid-samples skip --output-dir saved_model/wsm --resume saved_model/wsm/last.pth --epochs 300
```

`--epochs 300` is the **total** epoch count. Resume restores optimizer, scheduler,
best metric, RNG and loader states, and verifies data hashes, CSV rows, label
format, invalid-sample policy and split settings. CPU split-resume equivalence
was checked exactly; other hardware/CUDA kernels may differ. Existing run
directories are protected from accidental overwrite.

For fine-tuning or older weights, use `--init-weights` with a new output directory.
Old full checkpoints from before the native-data adapter should also be used as
weight initialization, since the data-reading/split configuration has changed.
Only load checkpoints from trusted sources.

## Validation and inference compatibility

Heatmap selection minimizes held-out BCE+Focal loss. Detector validation matches
the highest-scoring prediction to each vertebral tag and averages binary mask
IoU, counting missing tags as zero (score threshold 0.05, mask threshold 0.5).
The minimized detector objective is `1 - mean mask IoU`. This diagnostic is
**not COCO AP or the paper's final Cobb-angle SMAPE**.

The new detector is torchvision **Mask R-CNN ResNet50-FPN** with torchvision's
default normalization. The original is GluonCV/MXNet ResNet50-v1b with different
normalization and weights; `.pth` and `.params` are not interchangeable.

```python
from predict_detector import TorchvisionSpineDetector

detector = TorchvisionSpineDetector(
    "saved_model/maskrcnn/best.pth",
    device="cuda",
    wsm_checkpoint="saved_model/wsm/best_weights.pth",
)
masks, boxes, centers, tags = detector.forward(image)
```

Input is an `HxW` or `1xHxW` grayscale NumPy image. Outputs use the original
input resolution, with tags mapped to 0–16 and detections ordered by vertical
box center. Use `already_masked=True` if the legacy caller has already multiplied
the image by WSM. The adapter does not accept the legacy visualization/image-ID
positional arguments. Integration into the historical script is manual.

## Verification

```powershell
python -m unittest discover -s tests -v
```

Eleven synthetic regression tests cover all network heads, real detector
backpropagation, Windows workers, native/CSV pairing, audits, group splitting,
checkpoint loading and exact CPU resume. The six stages also completed one real
training and validation batch at a reduced 96×64 resolution. No full retraining,
GPU validation or final SMAPE evaluation has been performed.
