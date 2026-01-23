"""
Render spec schema definitions.

Order of contents:
- RenderSpecMeta
- RenderSpec
"""

from dataclasses import dataclass
from .layers import Layer


@dataclass
class RenderSpecMeta:
    """
    Per-view metadata for a RenderSpec.
    - frame_id/timestamp: time identity for the frame
    - dataset/sequence/scene/split/weather: optional dataset context
    - view_name: name of the view builder
    - sensor_ids: sensors contributing to the view
    - source_keys: overlay sources included (e.g., gt, pred runs)

    Example:
      frame_id=42
      timestamp=1678901234.5
      dataset="nuscenes"
      sequence_id="scene-0101"
      scene=None
      split="val"
      weather=None
      view_name="BEVView"
      sensor_ids=["lidar_top", "radar_front"]
      source_keys=["gt", "pred:run1"]
    """

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
    """
    Container for a single view's renderable layers.
    - title: human-readable view title
    - coord_frame: reference frame for layers (sensor/ego/world)
    - layers: list of Layer objects
    - meta: view-level metadata

    Example:
      title="cam_front"
      coord_frame="sensor:cam_front"
      layers=[RasterLayer(...), Box2DLayer(...)]
      meta=RenderSpecMeta(...)
    """

    title: str
    coord_frame: str | None  # For sensor-space or ego/world views, set explicit string
    layers: list[Layer]
    meta: RenderSpecMeta
