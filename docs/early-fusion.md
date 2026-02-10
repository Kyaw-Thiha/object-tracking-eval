# Early Fusion Models

Early fusion = fusing multiple sensors (camera + radar, camera + IR, etc.) at the **data level**, before detection. Work is primarily done in producers.

## File Checklist

| File | Location | Purpose |
|------|----------|---------|
| Producer | `src/data/producers/` | Fuse sensor data, build input for detector |
| Detector | `src/model/det/` | Detection head (optional, if custom needed) |
| Factory | `src/model/factory/` | Wrapper with `infer_with_context` |
| Config | `src/configs/` | Model config |

## Data Flow

```
Standard (single sensor):
  Camera → Producer → Image → Detector → Boxes

Early Fusion (multi-sensor):
  Camera ─┐
  Radar ──┼→ Producer → Fused Data → Detector → Boxes
  IR ─────┘
```

## Interface Contract

```python
def infer_with_context(self, imgs: Tensor, targets=None):
    """
    targets: List[dict] - contains fused sensor data from producer
      (e.g., proposals, embeddings, feature maps, metadata)
    """
```

The pipeline calls `infer_with_context` when the detector has this method, passing fused data from the producer.

## Factory Registration

1. Create `src/model/factory/<name>.py`
2. Export a `factory(device, checkpoint_path, config_path)` function
3. Return model instance with `infer_with_context` method

## Verification

```bash
python src/evaluation_pipeline.py --model <factory_name> --config <config> ...
```

## Reference Implementation (RRPN)

- Producer: `src/data/producers/rrpn.py`
- Factory: `src/model/factory/cascade_rcnn_rrpn.py`
- Config: `src/configs/cascade_rcnn/`
- Consolidated usage doc: `docs/rrpn.md`
