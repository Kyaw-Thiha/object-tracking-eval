from dataclasses import dataclass


@dataclass
class RenderSpecMeta:
    frame_id: int
    timestamp: float
    dataset: str | None
    sequence_id: str | None
    scene: str | None
    split: str | None
    weather: str | None
    view_name: str  # e.g., "CameraView", "RadarGridView", "BEVView"
    sensor_ids: list[str]  # sensors contributing to this view
    source_keys: list[str]  # e.g., ["gt", "pred:run1"]


@dataclass
class RenderSpec:
    title: str
    coord_frame: str | None  # For sensor‑space or ego/world views, set explicit string
    layers: list[Layer]
    meta: RenderSpecMeta
