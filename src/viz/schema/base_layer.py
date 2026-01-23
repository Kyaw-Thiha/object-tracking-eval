from dataclasses import dataclass


@dataclass
class LayerMeta:
    source: str  # "gt" | "pred:<run_id>" | "det:<name>"
    sensor_id: str | None  # "cam_front", "radar_rear", None for fused layers
    kind: str  # "image", "grid", "pc", "det", "track", "bbox2d", "bbox3d"
    coord_frame: str | None  # overrides RenderSpec.coord_frame if needed
    timestamp: float | None  # optional layer time tag


@dataclass
class LayerStyle:
    color: tuple[float, float, float] | None
    alpha: float  #  = 1.0
    line_width: float  # = 1.0
    point_size: float  # = 3.0
    colormap: str | None  # (e.g., "viridis")
    palette: dict[int, tuple[float, float, float]] | None  # (for class‑wise colors)


@dataclass
class Layer:
    name: str
    visible: bool
    style: LayerStyle
    meta: LayerMeta
