# NuScenes Viz Entrypoints

Quick commands to visualize NuScenes with the viz stack.

## Common Args
- `--dataset-path`: path to NuScenes root (default: `data/nuScenes`)
- Frame selection (choose one):
  - `--index <int>`
  - `--scene <scene-name> --frame-id <int>`
- `--source-key`: overlay source (default: `gt`)
- Radar GT overlays:
  - `--show-gt-centers`: overlay GT box centers (radar frame / RA grid)
  - `--show-gt-footprints`: overlay GT box footprints (radar frame / RA grid)
- Sequence controls:
  - `--end-index <int>`: inclusive end index (defaults to last frame)
  - `--step <int>`: stride between frames (default: 1)
  - `--play-interval <float>`: seconds per frame when playing (default: 0.2)
- Keybindings:
  - `open3d`: A/D step frames, Space toggles play
  - `napari`: Left/Right arrows step frames
  - `plotly`: slider + Play/Pause buttons

---

## Single-Sensor Views

### Camera
```bash
python -m src.viz.entrypoints.nuscenes.single.camera \
  --dataset-path data/nuScenes \
  --index 0 \
  --end-index 20 \
  --play-interval 0.2 \
  --sensor-id CAM_FRONT \
  --backend napari
```

### Radar Grid (RA/RD)
```bash
python -m src.viz.entrypoints.nuscenes.single.radar_grid \
  --dataset-path data/nuScenes \
  --index 0 \
  --end-index 20 \
  --play-interval 0.2 \
  --sensor-id RADAR_FRONT \
  --grid-name RA \
  --display polar \
  --show-gt-centers \
  --show-gt-footprints \
  --backend plotly
```

### Radar Point Cloud
```bash
python -m src.viz.entrypoints.nuscenes.single.radar_point \
  --dataset-path data/nuScenes \
  --index 0 \
  --end-index 20 \
  --play-interval 0.2 \
  --sensor-id RADAR_FRONT \
  --value-key doppler \
  --show-gt-centers \
  --show-gt-footprints \
  --backend plotly
```

### LiDAR
```bash
python -m src.viz.entrypoints.nuscenes.single.lidar \
  --dataset-path data/nuScenes \
  --index 0 \
  --end-index 20 \
  --play-interval 0.2 \
  --sensor-id LIDAR_TOP \
  --backend open3d
```

---

## Fusion Views

### BEV (LiDAR + Radar)
```bash
python -m src.viz.entrypoints.nuscenes.fusion.bev \
  --dataset-path data/nuScenes \
  --index 0 \
  --end-index 20 \
  --play-interval 0.2 \
  --sensor-ids LIDAR_TOP,RADAR_FRONT \
  --source-keys gt \
  --backend open3d
```

---

## Multi-Panel Views (Plotly)

### Camera + BEV
```bash
python -m src.viz.entrypoints.nuscenes.multi_panel.cam_bev \
  --dataset-path data/nuScenes \
  --index 0 \
  --end-index 20 \
  --play-interval 0.2 \
  --camera-id CAM_FRONT \
  --sensor-ids LIDAR_TOP,RADAR_FRONT \
  --source-key gt \
  --bev-max-points 20000 \
  --use-webgl
```

### Radar Grid + BEV
```bash
python -m src.viz.entrypoints.nuscenes.multi_panel.radar_bev \
  --dataset-path data/nuScenes \
  --index 0 \
  --end-index 20 \
  --play-interval 0.2 \
  --sensor-id RADAR_FRONT \
  --grid-name RA \
  --display pixel \
  --source-key gt \
  --show-gt-centers \
  --show-gt-footprints \
  --bev-max-points 20000 \
  --use-webgl
```

### 4-Panel (Camera + Radar Grid + BEV)
```bash
python -m src.viz.entrypoints.nuscenes.multi_panel.four_panel \
  --dataset-path data/nuScenes \
  --index 0 \
  --end-index 20 \
  --play-interval 0.2 \
  --camera-id CAM_FRONT \
  --radar-id RADAR_FRONT \
  --grid-name RA \
  --display pixel \
  --source-key gt \
  --show-gt-centers \
  --show-gt-footprints \
  --bev-max-points 20000 \
  --use-webgl
```

---

## Notes
- Backends:
  - `open3d` is recommended for LiDAR/BEV 3D views.
  - `napari` is best for camera images and 2D overlays.
  - `plotly` works well for radar grids and multi-panel layouts.
- Plotly multi-panel scripts currently render Plotly only.
- Multi-panel speed tips: `--bev-max-points` reduces point count; `--use-webgl` can improve performance but may drop BEV points during animation on some systems.
- Radar GT overlays are supported for RA grids; RD requires GT doppler to be available.

---

## Troubleshooting
- If a window does not appear, ensure the backend is installed (`open3d`, `napari`, `plotly`).
- If NuScenes data is not found, double-check `--dataset-path`.
- For polar radar grids, try `--display pixel` if your backend does not support polar rendering.
