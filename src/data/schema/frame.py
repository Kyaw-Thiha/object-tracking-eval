"""
Frame schema definitions.

Order of contents:
- FrameMeta
- SensorFrame
- Frame (top-level container for sensors + overlays)
"""

from dataclasses import dataclass
from typing import Optional, Union

from .radar import RadarSensorFrame
from .image import ImageSensorFrame
from .lidar import LidarSensorFrame
from .overlay import OverlaySet


@dataclass()
class FrameMeta:
    """
    Per-frame context metadata (optional).
    - sequence_id: video/scene identifier
    - dataset: dataset name (e.g., "radiate", "mot17")
    - scene: scene name/location
    - split: "train" / "val" / "test"
    - weather: optional conditions tag

    Example:
      sequence_id="sequence_01"
      dataset="radiate"
      scene="kumpulan_park"
      split="train"
      weather="rain"
    """

    sequence_id: Optional[str]  # video/scene id
    dataset: Optional[str]  # "radiate", "mot17", ...
    scene: Optional[str]  # scene name or location
    split: Optional[str]  # train/val/test
    weather: Optional[str]  # if provided


@dataclass()
class SensorFrame:
    """
    Wrapper for a single sensor modality at a frame.
    - kind: "radar" / "image" / "lidar" / ...
    - data: modality-specific sensor frame data
    """

    kind: str
    data: Union[RadarSensorFrame, ImageSensorFrame, LidarSensorFrame]


@dataclass()
class Frame:
    """
    Top-level data container for a single time step.
    - timestamp: time in seconds (float)
    - frame_id: integer index within a sequence
    - sensors: mapping from sensor_id -> SensorFrame
    - overlays: optional annotations (GT/predictions)
    - meta: optional frame-level context

    Example:
      timestamp=1678901234.5
      frame_id=42
      sensors={
        "radar_front": SensorFrame(kind="radar", data=RadarSensorFrame(...)),
        "cam_front": SensorFrame(kind="image", data=ImageSensorFrame(...)),
      }
      overlays=None
      meta=FrameMeta(sequence_id="seq_01", dataset="radiate", scene=None, split="train", weather=None)
    """

    timestamp: float
    frame_id: int
    sensors: dict[str, SensorFrame]
    overlays: Optional[OverlaySet] = None
    meta: Optional[FrameMeta] = None
