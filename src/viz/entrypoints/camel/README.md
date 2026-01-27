# CAMEL Viz Entrypoints

Quick commands to visualize CAMEL with the viz stack.

## Common Args
- `--dataset-path`: path to CAMEL root (default: `data/camel_dataset`)
- `--ann-file`: COCO annotation file (default: `annotations/half-train_cocoformat.json`)
- `--split`: dataset split folder for images (default: `train`)
- Sequence controls:
  - `--end-index <int>`: inclusive end index (defaults to last frame)
  - `--step <int>`: stride between frames (default: 1)
  - `--play-interval <float>`: seconds per frame when playing (default: 0.2)
- `--source-key`: overlay source (default: `gt`)
- Keybindings:
  - `napari`: Left/Right arrows step frames
  - `plotly`: slider + Play/Pause buttons

---

## Single-Sensor Views

### Camera
```bash
python -m src.viz.entrypoints.camel.single.camera \
  --dataset-path data/camel_dataset \
  --ann-file annotations/test_cocoformat_half.json \
  --split test_half \
  --index 0 \
  --end-index 200 \
  --play-interval 0.2 \
  --backend napari
```

---

## Notes
- Backends:
  - `napari` is best for camera images and 2D overlays.
  - `plotly` works well for quick interactive playback.
- If a window does not appear, ensure the backend is installed (`napari`, `plotly`).
- If CAMEL data is not found, double-check `--dataset-path`, `--ann-file`, and `--split`.
