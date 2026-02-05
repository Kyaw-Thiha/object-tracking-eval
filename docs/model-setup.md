# Model Setup

This repo uses MMDet/MMTrack as a model zoo (configs + weights) and runs custom
training/evaluation pipelines. The model layer is organized around detector
implementations, inference wrappers, and trackers.

## Overview

- `src/model/det/`: MMDet-style detector implementations (e.g., probabilistic YOLOX).
- `src/model/losses/`: custom losses registered with MMDet.
- `src/model/factory/`: detector wrappers used by the evaluation pipeline.
- `src/model/tracker/`: custom trackers used by the evaluation pipeline.
- `src/model/kalman_filter_*.py`: shared Kalman filter utilities.

## Detector Implementations

Detector code lives under `src/model/det/` and follows MMDet conventions. These
modules register `DETECTORS` and `HEADS` so they can be constructed from MMDet
configs in `src/configs/`.

Example paths:
- `src/model/det/yolox/`
- `src/model/det/bayesod/`

## Detector Factories (Evaluation)

The evaluation pipeline loads a detector through a factory module in
`src/model/factory/`. Each factory must expose:

- `factory(device: str) -> nn.Module` that returns a wrapper model.
- The wrapper must implement:
  - `infer(imgs: Tensor) -> (batch_bboxes, batch_labels, batch_covs)`
  - `get_classes() -> List[str]`

Factories are responsible for:
- Building the detector from an MMDet config.
- Loading checkpoint weights.
- Converting detector outputs to the standard inference format.

Example factory modules:
- `src/model/factory/prob_yolox.py`
- `src/model/factory/yolox_noise.py`
- `src/model/factory/yolox_identity_covs.py`

## MMDet Config + Checkpoint Loading

`src/model/factory/utils.py` provides shared helpers:

- `load_detector_from_checkpoint(config_path, checkpoint_path, device)`
  - Loads the MMDet config.
  - Builds the detector.
  - Loads weights (handles common key prefixes).
  - Moves the model to the requested device.

- `get_bboxes_from_detector(detector, imgs)`
  - Runs `extract_feat` and `bbox_head`.
  - Uses the head’s `get_bboxes()` to produce:
    - bounding boxes (N, 5)
    - covariance matrices (N, 4, 4)
    - labels (N,)

## Trackers

All trackers live under `src/model/tracker/` and are instantiated directly by
the evaluation pipeline.

Examples:
- `ProbabilisticByteTracker`
- `ProbabilisticOCSORTTracker`
- `UncertaintyTracker`

## Training

Training uses the custom loop in `src/train.py`:

- Builds a detector from an MMDet config (`src/configs/`).
- Optionally loads a checkpoint for initialization.
- Uses a dataloader factory from `src/data/dataloaders/`.
- Runs a PyTorch training loop and saves checkpoints.

For usage, see the `README.md` training example.
