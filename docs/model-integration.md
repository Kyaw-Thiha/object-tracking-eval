# Model Integration Guide

This doc explains how to integrate a new tracking model into the custom
evaluation/training pipeline. It covers detector-only models, detector+tracker
pipelines, and end-to-end tracking models, plus uncertainty and future
fusion models.

## Integration Paths (How-To)

Pick the path that matches your model and follow the checklist.

### 1) Detector-only (pipeline tracker)

Use this when your model only does detection and you want to use an existing
tracker from `src/model/tracker/`.

How-to:
1. Implement or reuse a detector under `src/model/det/` (MMDet-style).
2. Create a factory wrapper in `src/model/factory/` with `factory(device)` and
   the `infer()` / `get_classes()` interface.
3. Point `src/evaluation_pipeline.py` at the factory via `--model_factory`.
4. Choose a tracker via `--tracker` (e.g., `probabilistic_byte_tracker`).

Checklist:
- `infer()` returns `(bboxes, labels, covs)` per image.
- Covariances are `(N, 4, 4)`.

### 2) Detector + custom tracker (pipeline loop)

Use this when you need custom association logic or uncertainty-aware tracking.

How-to:
1. Add your tracker to `src/model/tracker/` and match the `track(...)` signature.
2. Ensure the tracker returns `track_bboxes, track_bbox_covs, track_labels, track_ids`.
3. (Optional) Add config args in `evaluation_pipeline.py` if your tracker has
   custom parameters.
4. Run the pipeline with `--tracker <your_tracker_name>`.

Checklist:
- `track()` consumes `bboxes`, `bbox_covs`, `labels`, and returns IDs.
- Output format matches the pipeline’s expectations.

### 3) End-to-end tracker (model outputs tracks)

Use this when the model produces tracks directly (IDs + boxes), and you don’t
want the built-in tracker.

How-to:
1. Wrap your model in a factory under `src/model/factory/`.
2. Implement `infer()` to return tracked outputs in the pipeline’s format
   (boxes, covs, labels, ids).
3. In `evaluation_pipeline.py`, bypass the tracker step and treat model output
   as final tracks (a small conditional branch is usually needed).

Checklist:
- You still emit `track_ids` per detection.
- Output is per frame and consistent with MOT-style evaluation.

Note: this path usually requires a small change in the pipeline to skip tracker
logic. If you want, we can add a flag like `--end_to_end_tracker` later.

## Required Interfaces

### Detector Wrapper (Evaluation)

All detector wrappers live under `src/model/factory/` and must expose:

- `factory(device: str) -> nn.Module`
- The returned model must implement:
  - `infer(imgs: Tensor) -> (batch_bboxes, batch_labels, batch_covs)`
  - `get_classes() -> List[str]`

Output conventions:

- `batch_bboxes`: list of tensors, each `(N, 5)` as `[x1, y1, x2, y2, score]`
- `batch_labels`: list of tensors, each `(N,)`
- `batch_covs`: list of tensors, each `(N, 4, 4)`

If your model does not produce covariance, return identity or a fixed diagonal
covariance (see existing factories for patterns).

### Tracker Interface

Trackers live under `src/model/tracker/` and are instantiated directly by
`src/evaluation_pipeline.py`.

A tracker must support:

- `track(img, img_metas, model, bboxes, bbox_covs, labels, frame_id)`
- Returns: `track_bboxes, track_bbox_covs, track_labels, track_ids`

Output conventions:

- `track_bboxes`: `(N, 5)` `[x1, y1, x2, y2, score]`
- `track_bbox_covs`: `(N, 4, 4)`
- `track_labels`: `(N,)`
- `track_ids`: `(N,)`

## Where to Put Code

- Detector implementation: `src/model/det/`
- Detector wrapper (factory): `src/model/factory/`
- Tracker implementation: `src/model/tracker/`
- Data loaders (if needed): `src/data/dataloaders/`

## Uncertainty Support

- The pipeline expects covariance as `Nx4x4` per frame.
- If your model predicts uncertainty, map it into a 4D box covariance matrix.
- If it does not, use identity or a fixed diagonal (see `yolox_identity_covs.py`).

## Fusion Models (Early / Mid / Late)

Fusion models will likely require extending data ingestion:

- **Early fusion**: merge modalities before the detector.
  - Suggested: implement fusion logic inside a new Producer under
    `src/data/producers/` and have dataloaders return fused tensors.
  - Example: `src/data/producers/rrpn.py` generates region proposals from radar
    detections projected to camera coordinates.
  - See `docs/rrpn.md` for the RRPN usage and command reference.
  - For RRPN probabilistic training specifics (NLL heads and troubleshooting),
    see `docs/rrpn.md`.

- **Mid fusion**: fuse features inside the model.
  - Suggested: implement in `src/model/det/` with multiple backbones/streams.

- **Late fusion**: fuse detections or tracks after inference.
  - Suggested: implement in a custom tracker or a post-processing step in
    the evaluation pipeline.

If fusion models require additional metadata, extend the `Frame` schema or
Producer outputs in `src/data/schema/` and `src/data/producers/`.

## Config + Checkpoints

- Use MMDet-style configs under `src/configs/`.
- Load weights via `load_detector_from_checkpoint` in `src/model/factory/utils.py`.

## Testing and Sanity Checks

- Run `src/evaluation_pipeline.py` on a small subset to confirm output format.
- Use `--annotate-videos` to visually validate IDs, boxes, and covariance.

### Smoke Pipeline

Use `src/smoke_pipeline.py` for fast preflight checks before full eval/training runs.

Eval smoke:

```bash
python src/smoke_pipeline.py eval \
  --dataloader_factory <your_factory> \
  --model_factory <your_model_factory> \
  --device cpu \
  --num_batches 1 \
  --with_context auto \
  --tracker uncertainty_tracker \
  --strict
```

Train smoke:

```bash
python src/smoke_pipeline.py train \
  --dataloader_factory <your_train_factory> \
  --config <config.py> \
  --device cpu \
  --num_batches 1 \
  --lr 1e-4 \
  --strict
```
