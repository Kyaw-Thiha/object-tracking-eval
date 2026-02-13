# RCBEVDet Component Selection

## Recommended Configuration

| # | Component | Selected Model |
|---|-----------|----------------|
| **1** | Image Backbone | **ResNet-50** |
| **2** | Image Neck | **CustomFPN** |
| **3** | View Transformer | **LSSViewTransformerBEVDepth** |
| **4** | Image BEV Encoder Backbone | **CustomResNet** |
| **5** | Image BEV Encoder Neck | **FPN_LSS** |
| **6-7** | Radar Voxel Encoder | **RadarBEVNet** |
| **8** | Radar Middle Encoder | **PointPillarsScatterRCS** |
| **9** | Radar BEV Backbone | **SECOND** |
| **10** | Radar BEV Neck | **SECONDFPN** |
| **11** | Fusion Module | **CAMF** (MSDeformAttn + RadarConvFuser) |
| **12** | Detection Head | **CenterHead** |
| **-** | Complete Detector | **BEVDepth4D_RC** |

## Configuration Parameters

| Parameter | Value |
|-----------|-------|
| Image Size | 256 × 704 |
| BEV Size | 128 × 128 |
| Radar Sweeps | 9 (8 past + 1 current) |
| Radar Features | 64-dim |
| Camera Features | 256-dim |
| Fusion Heads | 8 |
| Fusion Points | 8 |
| Training Epochs | 12 |

## Lightweight Alternative

| Component | Alternative Model |
|-----------|------------------|
| Image Backbone | ResNet-18 or DLA-34 |
| BEV Size | 64 × 64 |
| Radar Sweeps | 4 frames |

## Compile Ops

Run from repo root:

```bash
python src/model/det/rcbevdet/ops/deformattn/setup.py build_ext --inplace && \
python src/model/det/rcbevdet/ops/bev_pool_v2/setup.py build_ext --inplace
```

## What RCBEVDet Provides

- `RCBEVDetProducer` for NuScenes multi-camera + multi-radar inputs (paper-aligned defaults).
- `rcbevdet_nuscenes_factory` dataloader with custom collate for variable radar points.
- `model.factory.rcbevdet` wrapping real `BEVDepth4D_RC`.
- Native 3D outputs as primary path:
  - `boxes_3d`
  - `scores_3d`
  - `labels_3d`
  - `velocities` (when available)
- Optional 2D projection fallback for legacy 2D tracker compatibility.

## Quick Commands

### 1) Data Path Smoke (dataloader only)

```bash
python - <<'PY'
from data.dataloaders.rcbevdet_nuscenes_factory import factory
dl = factory()
imgs, targets = next(iter(dl))
print("imgs:", tuple(imgs.shape))
print("batch targets:", len(targets))
print("target keys:", sorted(targets[0].keys()))
PY
```

### 2) Model Path Smoke (3D detector only)

```bash
python src/smoke_pipeline.py eval \
  --dataloader_factory rcbevdet_nuscenes_factory \
  --model_factory rcbevdet \
  --device cpu \
  --num_batches 1 \
  --output_mode 3d \
  --tracker none \
  --strict
```

### 3) 3D Eval Smoke (no checkpoint)

```bash
python -X faulthandler src/evaluation_pipeline.py \
  --dataloader_factory rcbevdet_nuscenes_factory \
  --dataset_dir data/nuScenes \
  --model_factory rcbevdet \
  --smoke_eval \
  --bev_pool_backend torch \
  --tracker none \
  --output_mode 3d \
  --device cuda \
  --output_dir outputs/nuscenes_rcbevdet_3d_det_smoke \
  --eval_result_dir evaluation_results/nuscenes_rcbevdet_3d_det_smoke \
  --plot_save_path plots/nuscenes_rcbevdet_3d_det_smoke.png
```

### 4) Full 3D Eval (detector only, real weights)

```bash
python -X faulthandler src/evaluation_pipeline.py \
  --dataloader_factory rcbevdet_nuscenes_factory \
  --dataset_dir data/nuScenes \
  --model_factory rcbevdet \
  --checkpoint_path /path/to/rcbevdet.ckpt \
  --tracker none \
  --output_mode 3d \
  --device cuda \
  --output_dir outputs/nuscenes_rcbevdet_3d_det \
  --eval_result_dir evaluation_results/nuscenes_rcbevdet_3d_det \
  --plot_save_path plots/nuscenes_rcbevdet_3d_det.png
```

### 5) Full 3D Eval (detector + 3D tracker, real weights)

```bash
python -X faulthandler src/evaluation_pipeline.py \
  --dataloader_factory rcbevdet_nuscenes_factory \
  --dataset_dir data/nuScenes \
  --model_factory rcbevdet \
  --checkpoint_path /path/to/rcbevdet.ckpt \
  --tracker rcbevdet_3d_tracker \
  --output_mode 3d \
  --device cuda \
  --output_dir outputs/nuscenes_rcbevdet_3d_track \
  --eval_result_dir evaluation_results/nuscenes_rcbevdet_3d_track \
  --plot_save_path plots/nuscenes_rcbevdet_3d_track.png
```

### 5) Optional Legacy 2D Fallback Smoke

```bash
python src/smoke_pipeline.py eval \
  --dataloader_factory rcbevdet_nuscenes_factory \
  --model_factory rcbevdet \
  --device cpu \
  --num_batches 1 \
  --output_mode 2d \
  --with_context on \
  --tracker uncertainty_tracker \
  --strict
```

## Training Commands

### Train Smoke

```bash
python src/smoke_pipeline.py train \
  --dataloader_factory rcbevdet_nuscenes_factory \
  --config src/configs/rcbevdet/rcbevdet_r50_nuscenes_train.py \
  --device cpu \
  --num_batches 1 \
  --lr 1e-4 \
  --strict
```

### Full Training

```bash
python src/train.py \
  --config src/configs/rcbevdet/rcbevdet_r50_nuscenes_train.py \
  --work-dir results/rcbevdet_r50_train \
  --dataloader-factory rcbevdet_nuscenes_factory \
  --device cuda \
  --epochs 12 \
  --batch-size 2 \
  --lr 1e-4
```

## Prerequisites

- CUDA/C++ ops (`deformattn`, `bev_pool_v2`) are optional for correctness but recommended for performance; compile them before large-scale training/eval runs.
- RCBEVDet factory is fail-fast for missing checkpoint/config unless `--smoke_eval` is explicitly set.
- Pipeline factories call `factory(device=...)`; ensure checkpoint/config paths are resolved in `src/model/factory/rcbevdet.py` for your environment before running smoke/eval.

## Files

- `src/data/producers/rcbevdet.py`
- `src/data/adapters/nuscenes_rc.py`
- `src/data/dataloaders/rcbevdet_nuscenes_factory.py`
- `src/model/factory/rcbevdet.py`
- `src/model/tracker/rcbevdet_3d_tracker.py`
- `src/data/schema/prediction_3d.py`
- `src/model/det/rcbevdet/`

## CUDA Caveats

- RCBEVDet supports two modes:
  - Compiled CUDA ops (`deformattn`, `bev_pool_v2`) for best speed.
  - PyTorch fallback when CUDA extensions are unavailable.
- Fallback mode is expected to be slower (roughly 2-5x on affected ops) but is valid for smoke tests, debugging, and CI.
- If compiling CUDA ops, keep versions aligned:
  - `torch==1.13.1`
  - `mmcv-full==1.7.2`
  - CUDA 11.7 toolchain
  - GCC 11.x (commonly 11.2)
- Use `scripts/compile_rcbevdet_cuda.sh` from repo root to compile and generate logs:
  - `src/model/det/rcbevdet/ops/deformattn/compile_deformattn.log`
  - `src/model/det/rcbevdet/ops/bev_pool_v2/compile_bev_pool.log`
- If CUDA compilation fails, continue with fallback mode first and only optimize later when performance is a bottleneck.
