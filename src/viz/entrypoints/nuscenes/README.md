# NuScenes Viz Entrypoints

Quick commands to visualize NuScenes with the viz stack.

## Common Args
- `--dataset-path`: path to NuScenes root (default: `data/nuScenes`)
- Frame selection (choose one):
  - `--index <int>`
  - `--scene <scene-name> --frame-id <int>`
- `--source-key`: overlay source (default: `gt`)

---

## Single-Sensor Views

### Camera
```bash
python -m src.viz.entrypoints.nuscenes.single.camera \
  --dataset-path data/nuScenes \
  --index 0 \
  --sensor-id CAM_FRONT \
  --backend napari
```

### Radar Grid (RA/RD)
```bash
python -m src.viz.entrypoints.nuscenes.single.radar_grid \
  --dataset-path data/nuScenes \
  --index 0 \
  --sensor-id RADAR_FRONT \
  --grid-name RA \
  --display polar \
  --backend plotly
```

### Radar Point Cloud
```bash
python -m src.viz.entrypoints.nuscenes.single.radar_point \
  --dataset-path data/nuScenes \
  --index 0 \
  --sensor-id RADAR_FRONT \
  --value-key doppler \
  --backend plotly
```

### LiDAR
```bash
python -m src.viz.entrypoints.nuscenes.single.lidar \
  --dataset-path data/nuScenes \
  --index 0 \
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
  --camera-id CAM_FRONT \
  --sensor-ids LIDAR_TOP,RADAR_FRONT \
  --source-key gt
```

### Radar Grid + BEV
```bash
python -m src.viz.entrypoints.nuscenes.multi_panel.radar_bev \
  --dataset-path data/nuScenes \
  --index 0 \
  --sensor-id RADAR_FRONT \
  --grid-name RA \
  --display pixel \
  --source-key gt
```

### 4-Panel (Camera + Radar Grid + BEV)
```bash
python -m src.viz.entrypoints.nuscenes.multi_panel.four_panel \
  --dataset-path data/nuScenes \
  --index 0 \
  --camera-id CAM_FRONT \
  --radar-id RADAR_FRONT \
  --grid-name RA \
  --display pixel \
  --source-key gt
```

---

## Notes
- Backends:
  - `open3d` is recommended for LiDAR/BEV 3D views.
  - `napari` is best for camera images and 2D overlays.
  - `plotly` works well for radar grids and multi-panel layouts.
- Plotly multi-panel scripts currently render Plotly only.

---

## Troubleshooting
- If a window does not appear, ensure the backend is installed (`open3d`, `napari`, `plotly`).
- If NuScenes data is not found, double-check `--dataset-path`.
- For polar radar grids, try `--display pixel` if your backend does not support polar rendering.
