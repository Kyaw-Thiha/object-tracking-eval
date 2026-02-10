# RRPN + Cascade R-CNN (Standard and Probabilistic)

This is the single reference for RRPN-based Cascade R-CNN usage in this repo.
For generic early-fusion architecture guidance, see `docs/early-fusion.md`.

## What RRPN Provides

`RRPNProducer` generates camera-aligned proposal boxes from radar points and
passes them through batch targets to the detector factory wrapper.

```python
from src.data.adapters.nuscenes import NuScenesAdapter
from src.data.producers.rrpn import RRPNProducer

adapter = NuScenesAdapter(dataset_path="data/nuscenes")
producer = RRPNProducer(
    adapter,
    camera_id="CAM_FRONT",
    radar_ids=["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT"],
    alpha=20.0,  # S = alpha / d + beta
    beta=0.5,
)

sensors, target = producer[0]
# sensors["image"]: (3, H, W)
# target["proposals"]: (N, 4) [x1, y1, x2, y2]
# target["radar_meta"]: radar metadata (range, velocity, RCS, ...)
```

## Model Variants

### Standard Cascade R-CNN + RRPN

```python
from src.model.factory.cascade_rcnn_rrpn import factory

model = factory(
    device="cuda:0",
    checkpoint_path="path/to/cascade_rcnn_weights.pth",  # optional
)

batch_dets = model(imgs, targets)
bboxes, labels, covs = batch_dets
```

### Probabilistic Cascade R-CNN + RRPN

```python
from src.model.factory.cascade_rcnn_rrpn_prob import factory

model = factory(
    device="cuda:0",
    checkpoint_path="path/to/probabilistic_weights.pth",  # optional
)

batch_dets = model(imgs, targets)
bboxes, labels, covs = batch_dets
# covs: list[(N_i, 4, 4)] covariance matrices
```

## Quick Commands

### Evaluation Pipeline

`src/evaluation_pipeline.py` requires all args below.

Standard RRPN wrapper:

```bash
python -X faulthandler src/evaluation_pipeline.py \
  --dataloader_factory rrpn_nuscenes_factory \
  --dataset_dir data/nuScenes \
  --model_factory cascade_rcnn_rrpn \
  --tracker <uncertainty_tracker|probabilistic_byte_tracker|prob_ocsort_tracker> \
  --device cuda \
  --output_dir outputs/nuscenes_cascade_rcnn_rrpn_uncertainty_tracker \
  --eval_result_dir evaluation_results/nuscenes_cascade_rcnn_rrpn_uncertainty_tracker \
  --plot_save_path plots/nuscenes_cascade_rcnn_rrpn_uncertainty_tracker.png
```

Probabilistic RRPN wrapper:

```bash
python -X faulthandler src/evaluation_pipeline.py \
  --dataloader_factory rrpn_nuscenes_factory \
  --dataset_dir data/nuScenes \
  --model_factory cascade_rcnn_rrpn_prob \
  --tracker <uncertainty_tracker|probabilistic_byte_tracker|prob_ocsort_tracker> \
  --device cuda \
  --output_dir outputs/nuscenes_cascade_rcnn_rrpn_prob_uncertainty_tracker \
  --eval_result_dir evaluation_results/nuscenes_cascade_rcnn_rrpn_prob_uncertainty_tracker \
  --plot_save_path plots/nuscenes_cascade_rcnn_rrpn_prob_uncertainty_tracker.png
```

### Training Pipeline

Standard RRPN config:

```bash
python src/train.py \
  --config src/configs/cascade_rcnn/cascade_rcnn_r50_rrpn_nuscenes.py \
  --work-dir results/cascade_rrpn_train \
  --dataloader-factory rrpn_nuscenes_factory \
  --device cuda \
  --epochs 80 \
  --batch-size 4 \
  --lr 1e-4
```

Probabilistic RRPN config:

```bash
python src/train.py \
  --config src/configs/cascade_rcnn/cascade_rcnn_r50_rrpn_prob_nuscenes.py \
  --work-dir results/cascade_rrpn_prob_train \
  --dataloader-factory rrpn_nuscenes_factory \
  --device cuda \
  --epochs 80 \
  --batch-size 4 \
  --lr 1e-4
```

Optional resume from checkpoint (standard):

```bash
python src/train.py \
  --config src/configs/cascade_rcnn/cascade_rcnn_r50_rrpn_nuscenes.py \
  --checkpoint <path_to_checkpoint.pth> \
  --work-dir results/cascade_rrpn_train \
  --dataloader-factory rrpn_nuscenes_factory
```

Optional resume from checkpoint (probabilistic):

```bash
python src/train.py \
  --config src/configs/cascade_rcnn/cascade_rcnn_r50_rrpn_prob_nuscenes.py \
  --checkpoint <path_to_checkpoint.pth> \
  --work-dir results/cascade_rrpn_prob_train \
  --dataloader-factory rrpn_nuscenes_factory
```

### Smoke Checks

Use this for quick RRPN integration sanity checks before long runs.

Eval smoke (standard RRPN wrapper):

```bash
python src/smoke_pipeline.py eval \
  --dataloader_factory rrpn_nuscenes_factory \
  --model_factory cascade_rcnn_rrpn \
  --device cpu \
  --num_batches 1 \
  --with_context auto \
  --tracker uncertainty_tracker \
  --strict
```

Eval smoke (probabilistic RRPN wrapper):

```bash
python src/smoke_pipeline.py eval \
  --dataloader_factory rrpn_nuscenes_factory \
  --model_factory cascade_rcnn_rrpn_prob \
  --device cpu \
  --num_batches 1 \
  --with_context auto \
  --tracker uncertainty_tracker \
  --strict
```

Train smoke:

```bash
python src/smoke_pipeline.py train \
  --dataloader_factory rrpn_nuscenes_factory \
  --config src/configs/cascade_rcnn/cascade_rcnn_r50_rrpn_nuscenes.py \
  --device cpu \
  --num_batches 1 \
  --lr 1e-4 \
  --strict
```

Train smoke (probabilistic RRPN config):

```bash
python src/smoke_pipeline.py train \
  --dataloader_factory rrpn_nuscenes_factory \
  --config src/configs/cascade_rcnn/cascade_rcnn_r50_rrpn_prob_nuscenes.py \
  --device cpu \
  --num_batches 1 \
  --lr 1e-4 \
  --strict
```

## Probabilistic Training Notes

The probabilistic RRPN path is now a single end-to-end setup for both training
and evaluation.

- `ProbabilisticBBoxHead` predicts three outputs per RoI:
  `(cls_score, bbox_pred, bbox_cov)` where `bbox_cov` is log-variance.
- `ProbabilisticCascadeRoIHead` passes covariance into bbox loss during
  training and returns covariance-aligned detections at inference.
- `cascade_rcnn_rrpn_prob` is the single factory entrypoint for eval.

NLL intuition (diagonal covariance):

```text
loss ~= 0.5 * exp(-cov) * |bbox_pred - target| + 0.5 * cov
```

- Larger predicted covariance lowers the penalty on residual error.
- The `+ 0.5 * cov` term discourages over-inflated uncertainty.

Troubleshooting:

- NaN/instability:
  verify log-variance clamping and lower LR if needed.
- Covariance not learning:
  increase `loss_bbox.loss_weight` in the probabilistic bbox head config.
- Training too slow:
  keep `covariance_type='diagonal'` instead of full covariance.

## RRPN Parameters

- `camera_id`: camera sensor id (example: `CAM_FRONT`)
- `radar_ids`: list of radar sensors to fuse
- `alpha`: inverse-distance scale coefficient in `S = alpha / d + beta`
- `beta`: baseline scale offset in `S = alpha / d + beta`

## Files

- `src/data/producers/rrpn.py`: RRPN proposal producer
- `src/model/factory/cascade_rcnn_rrpn.py`: standard RRPN factory wrapper
- `src/model/factory/cascade_rcnn_rrpn_prob.py`: probabilistic RRPN wrapper
- `src/model/det/cascade_rcnn_rrpn/prob_bbox_head.py`: probabilistic bbox head
- `src/model/det/cascade_rcnn_rrpn/prob_cascade_roi_head.py`: probabilistic cascade RoI head
- `src/configs/cascade_rcnn/cascade_rcnn_r50_rrpn_nuscenes.py`: standard config
- `src/configs/cascade_rcnn/cascade_rcnn_r50_rrpn_prob_nuscenes.py`: probabilistic config

## Reference

Nabati & Qi, "RRPN: Radar Region Proposal Network for Object Detection in
Autonomous Vehicles", ICASSP 2019.
