# Data Architecture

This doc describes the data architecture used in this repo.

## Reasoning
The main reasoning behind the architecture is to be able to support all different types of dataset, as well as have the capabilites to support sensor fusion.
It also should be able to support `multi-view sensor fusion` which is a type of data not available in any public datasets that we know of, as of January 2026.

- Hence, we focus on extensibility towards the dataset layer using `Adapters`.
- Then, we use the Adapters to consolidate all these datasets into a single canonical format - a `Frame`.
- This `Frame` is then passed onto different `Producers` whose output is meant to be used by different tasks,
like by dataloaders for evaluation & training pipeline.

The architecture can be visualized as
```python
Dataset --> Adapters --> Frame (Canonical Form) --> Producers --> Specific Tasks (Training, Evaluation, Visualization, etc)
```

---

## API
### Schemas
All schemas live in `src/data/schema/`. The core types are:

- `FrameMeta`: optional, per-frame context (sequence id, dataset name, split, etc).
- `SensorFrame`: wrapper for a single sensor modality with a `kind` string and
  modality-specific payload.
- `Frame`: top-level container for timestamp, frame id, sensors, overlays, meta.

Conventions used in the current codebase:

- **Timestamp** is a float in seconds.
- **Frame id** is an integer index within a sequence.
- **Sensor ids** are free-form strings (e.g., `cam`, `cam_front`, `radar_front`).
- **Coordinate frames** are string tags like `sensor:cam_front`, `ego`, `world`.

Sensor schema highlights:

- `ImageSensorFrame`: `image` is HxW or HxWxC; `ImageMeta` includes intrinsics and
  a `sensor_pose_in_ego` SE(3) transform.
- `LidarSensorFrame`: `point_cloud` contains `xyz` (N,3) and optional per-point
  `features` (aligned with N).
- `RadarSensorFrame`: supports either `point_cloud` (xyz + features) or `grids`
  (range/azimuth/doppler products).

Overlay schema highlights:

- `OverlayMeta` is attached to every overlay object (coordinate frame, source, etc).
- `OverlaySet` groups overlay objects by source key, e.g. `"gt"` or `"pred:<run_id>"`.
- Supported overlay types include 2D/3D boxes, radar detections, and tracks.

### Adding a new Adapter
When you integrate a new dataset, you will need to create a new `Adapter` to convert it into `Frame`.
You should inherit from `BaseAdapter` inside `src/data/adapters/base.py`.
Then, you must define
- `index_frames(self) -> list[dict]` which returns dicts with at least:
```python
{
    "timestamp": <float seconds>,
    "frame_id": frame_idx,
    "meta": FrameMeta(...),  # optional but recommended
}
```
Any extra fields needed by `load_sensors` / `load_overlays` should also be present.

- `load_sensors(self, info: dict) -> dict[str, SensorFrame]` which returns a dictionary of shape `sensor_id: SensorFrame`
- `load_overlays(self, info: dict) -> OverlaySet | None` which returns an `OverlaySet` object

You should refer to existing implementations in
- `src/data/adapters/nuscenes.py`
- `src/data/adapters/camel.py`
- `src/data/adapters/coco.py` (shared COCO base adapter)

Notes from existing adapters:

- `NuScenesAdapter` uses `sensor_id` = NuScenes channel name and fills in
  `ego_pose_in_world` in metadata.
- `CocoBaseAdapter` maps dataset category ids to contiguous class ids and
  builds `Box2D` overlays in `sensor:cam` space.

### Adding a new Producer
To connect a `Frame` to a specific task, you will need to define a Producer.
You should inherit from `BaseProducer` inside `src/data/producers/base.py`.
Then, you must define
- `get_sensors(self, frame: Frame) -> dict[str, Any]`
- `preprocess(self, sensors: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]`
- `build_target(self, frame: Frame, meta: dict[str, Any]) -> dict[str, Any]`

You should refer to existing implementations in
- `src/data/producers/camel.py`

Producer contract:

- `__getitem__` returns `(sensors, target)`.
- `preprocess` can return additional `meta` (e.g., scale factor, padding).
- `build_target` should use `Frame.overlays` and `Frame.meta` if present.

---
