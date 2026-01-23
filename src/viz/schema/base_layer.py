"""
Base layer schema definitions.

Order of contents:
- LayerMeta
- LayerStyle
- Layer (base class for all layers)
"""

from dataclasses import dataclass, field


@dataclass
class LayerMeta:
    """
    Metadata attached to a single render layer.
    - source: label source (e.g., "gt" or "pred:<run_id>")
    - sensor_id: sensor association (None for fused layers)
    - kind: semantic layer type used for filtering/toggles
    - coord_frame: overrides RenderSpec.coord_frame if needed
    - timestamp: optional time tag for this layer

    Example:
      source="gt"
      sensor_id="cam_front"
      kind="bbox2d"
      coord_frame="sensor:cam_front"
      timestamp=1678901234.5
    """

    source: str  # "gt" | "pred:<run_id>" | "det:<name>"
    sensor_id: str | None  # "cam_front", "radar_rear", None for fused layers
    kind: str  # "image", "grid", "pc", "det", "track", "bbox2d", "bbox3d"
    coord_frame: str | None  # overrides RenderSpec.coord_frame if needed
    timestamp: float | None  # optional layer time tag


@dataclass
class LayerStyle:
    """
    Styling parameters for a layer.
    - color: RGB tuple in [0,1] or None to use default coloring
    - alpha: opacity in [0,1]
    - line_width: line thickness for boxes/lines
    - point_size: point size for scatter layers
    - colormap: named colormap for scalar values
    - palette: optional class_id -> RGB mapping

    Example:
      color=(0.2, 0.8, 0.2)
      alpha=0.9
      line_width=2.0
      point_size=3.0
      colormap="viridis"
      palette={0: (1.0, 0.0, 0.0), 1: (0.0, 1.0, 0.0)}
    """

    color: tuple[float, float, float] | None = None
    alpha: float = 1.0
    line_width: float = 1.0
    point_size: float = 3.0
    colormap: str | None = None  # (e.g., "viridis")
    palette: dict[int, tuple[float, float, float]] | None = None  # (for class-wise colors)


@dataclass(kw_only=True)
class Layer:
    """
    Base class for all renderable layers.
    - name: unique layer name for UI toggles
    - visible: whether the layer is shown by default
    - style: visual style settings
    - meta: standardized metadata for filtering

    Example:
      name="cam_front.image"
      visible=True
      style=LayerStyle()
      meta=LayerMeta(source="gt", sensor_id="cam_front", kind="image", coord_frame="sensor:cam_front", timestamp=1678901234.5)
    """

    name: str
    meta: LayerMeta
    visible: bool = True
    style: LayerStyle = field(default_factory=LayerStyle)
