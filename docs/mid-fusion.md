# Mid Fusion Models

Mid fusion = fusing multiple sensors (camera + radar, camera + lidar, etc.) at the **feature level**, inside the model. Work is split across producers (input packaging) and detector modules (fusion logic).

## File Checklist

| File | Location | Purpose |
|------|----------|---------|
| Adapter (optional) | `src/data/adapters/` | Extend dataset access for extra metadata (e.g., sweeps, poses, calibration) |
| Producer | `src/data/producers/` | Prepare per-sensor tensors and context metadata (no fusion here) |
| Dataloader Factory | `src/data/dataloaders/` | Build dataloader and custom collate for variable-length inputs |
| Detector Modules | `src/model/det/<model_name>/` | Implement mid-fusion architecture (backbone/neck/head/ops) |
| Factory | `src/model/factory/` | Wrapper with `factory(device, checkpoint_path, config_path, ...)` |
| Tracker (optional) | `src/model/tracker/` | Tracking module if model uses tracker path (e.g., 3D tracking) |
| Schema (optional) | `src/data/schema/` | Add output schema if introducing new output type (e.g., 3D detections) |
| Config | `src/configs/<model_name>/` | Model/training/inference config |

## Data Flow

```
Standard (single sensor):
  Camera → Producer → Image → Detector → Boxes

Mid Fusion (multi-sensor):
  Camera ─┐
  Radar ──┼→ Producer → Structured Inputs → Mid-Fusion Detector → Boxes/Tracks
  Lidar ──┘                             (fusion happens inside model)
```

## Interface Contract

```python
def infer_with_context(self, imgs: Tensor, targets=None):
    """
    targets: List[dict] containing per-sensor inputs + metadata
      (e.g., img_inputs, radar_points, img_metas, frame_id, video_id)
    """
```

If supporting native 3D outputs, also expose:

```python
def infer_with_context_3d(self, imgs: Tensor, targets=None):
    """
    returns: List[dict] with keys like boxes_3d, scores_3d, labels_3d
    """
```

Pipeline-facing model requirements:
- `get_classes() -> list[str]`
- `infer_with_context(...)` for 2D compatibility
- `infer_with_context_3d(...)` when using 3D output mode

## Factory Registration

1. Create `src/model/factory/<name>.py`
2. Export `factory(device, checkpoint_path=None, config_path=None, ...)`
3. Build and return wrapper instance implementing pipeline contract methods

## Pipeline Wiring Checklist

1. Add `src/data/dataloaders/<name>_factory.py` for dataset + producer wiring.
2. Ensure producer output keys match model wrapper expected target keys.
3. Add model factory module under `src/model/factory/`.
4. If adding tracker mode, export tracker from `src/model/tracker/__init__.py`.
5. If adding new output type (e.g., 3D), add schema under `src/data/schema/` and ensure evaluation path supports it.

## Verification

```bash
# 1) Data path smoke
python src/smoke_pipeline.py eval --dataloader_factory <name> --model_factory <name> --num_batches 1 ...

# 2) Model path smoke
python src/smoke_pipeline.py eval --dataloader_factory <name> --model_factory <name> --output_mode <2d|3d> --num_batches 1 ...

# 3) End-to-end evaluation smoke
python src/evaluation_pipeline.py --dataloader_factory <name> --model_factory <name> --output_mode <2d|3d> ...
```

## Reference Implementation (RCBEVDet)

- Adapter: `src/data/adapters/nuscenes_rc.py`
- Producer: `src/data/producers/rcbevdet.py`
- Dataloader: `src/data/dataloaders/rcbevdet_nuscenes_factory.py`
- Factory: `src/model/factory/rcbevdet.py`
- Tracker (3D): `src/model/tracker/rcbevdet_3d_tracker.py`
- Schema (3D): `src/data/schema/prediction_3d.py`
- Model-specific run commands and caveats: `docs/rcbevdet.md`
