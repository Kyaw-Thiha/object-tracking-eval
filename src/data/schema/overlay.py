from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass()
class OverlayMeta:
    """
    Metadata for overlay annotations.
    - coord_frame: reference frame for all coordinates in the overlay
    - source: label source (e.g., "gt" or "pred:<run_id>")
    - timestamp: optional time tag for this overlay
    - sensor_id: optional sensor association

    Example:
      coord_frame="ego"
      source="gt"
      timestamp=1678901234.5
      sensor_id="cam_front"
    """
    coord_frame: str  # "sensor:cam_front", "ego", "world", "bev", etc.
    source: str  # "gt", "pred:<run_id>", "detector:<name>", etc.
    timestamp: Optional[float]  # if overlay is time-specific
    sensor_id: Optional[str]  # if labels tied to a specific sensor


@dataclass()
class Box3D:
    """
    3D axis-aligned box with yaw in a named coordinate frame.
    - center_xyz: (3,) box center
    - size_lwh: (3,) size in length, width, height
    - yaw: rotation around +z in radians
    - class_id: integer class index
    - confidence/track_id/velocity_xyz: optional annotations

    Example:
      center_xyz=np.array([1.0, 2.0, 0.5], dtype=np.float32)
      size_lwh=np.array([4.0, 1.8, 1.5], dtype=np.float32)
      yaw=0.25
      class_id=2
      confidence=0.92
      track_id=7
      velocity_xyz=np.array([0.5, 0.0, 0.0], dtype=np.float32)
    """

    meta: OverlayMeta
    center_xyz: np.ndarray  # (3,) float32
    size_lwh: np.ndarray  # (3,) float32  (length, width, height)
    yaw: float  # radians (rotation around +z in coord_frame)
    class_id: int
    confidence: Optional[float]
    track_id: Optional[int]
    velocity_xyz: Optional[np.ndarray]


@dataclass()
class Box2D:
    """
    2D axis-aligned bounding box.
    - xyxy: (4,) [x1, y1, x2, y2] in pixels or grid coords
    - class_id: integer class index
    - confidence/track_id: optional annotations

    Example:
      xyxy=np.array([100.0, 50.0, 220.0, 180.0], dtype=np.float32)
      class_id=1
      confidence=0.88
      track_id=12
    """

    meta: OverlayMeta
    xyxy: np.ndarray  # (4,) float32: x1,y1,x2,y2 in pixels or grid coords
    class_id: int
    confidence: Optional[float]
    track_id: Optional[int]


@dataclass()
class OrientedBox2D:
    """
    2D oriented bounding box.
    - center_xyz: (3,) box center (xy used, z ignored if unused)
    - size_lwh: (3,) size in length, width, height (z can be 0)
    - yaw: rotation in radians
    - class_id: integer class index
    - confidence/track_id: optional annotations

    Example:
      center_xyz=np.array([120.0, 80.0, 0.0], dtype=np.float32)
      size_lwh=np.array([40.0, 20.0, 0.0], dtype=np.float32)
      yaw=-0.5
      class_id=3
      confidence=0.75
      track_id=21
    """

    meta: OverlayMeta
    center_xyz: np.ndarray  # (3,) float32
    size_lwh: np.ndarray  # (3,) float32  (length, width, height)
    yaw: float  # radians (rotation around +z in coord_frame)
    class_id: int
    confidence: Optional[float]
    track_id: Optional[int]


@dataclass()
class RadarPointDetections:
    """
    Cartesian radar detections with per-point features.
    - xyz: (N, 3) points in coord_frame
    - features: per-point arrays aligned with xyz (N,)
    - class_id/confidence: optional per-point labels

    Example:
      xyz=np.zeros((128, 3), dtype=np.float32)
      features={"doppler": np.zeros(128), "rcs": np.zeros(128), "snr": np.zeros(128)}
      class_id=None
      confidence=None
    """

    meta: OverlayMeta
    xyz: np.ndarray  # (N,3) in coord_frame (often sensor or ego)
    features: dict[str, np.ndarray]  # e.g. {"doppler": (N,), "rcs": (N,), "snr": (N,)}
    class_id: Optional[np.ndarray]
    confidence: Optional[np.ndarray]


@dataclass()
class RadarPolarDetections:
    """
    Polar radar detections.
    - range_m: (N,) ranges in meters
    - azimuth_rad/doppler_mps/amplitude: optional per-detection fields

    Example:
      range_m=np.array([12.3, 20.1], dtype=np.float32)
      azimuth_rad=np.array([0.1, -0.2], dtype=np.float32)
      doppler_mps=np.array([1.5, -0.3], dtype=np.float32)
      amplitude=np.array([42.0, 35.0], dtype=np.float32)
    """

    meta: OverlayMeta
    range_m: np.ndarray
    azimuth_rad: Optional[np.ndarray]
    doppler_mps: Optional[np.ndarray]
    amplitude: Optional[np.ndarray]


@dataclass()
class TrackState:
    """
    Single track state at a timestamp.
    - position_xyz: (3,) position in track coord_frame
    - velocity_xyz/yaw/covariance: optional kinematic info

    Example:
      timestamp=1678901234.5
      position_xyz=np.array([5.0, 1.2, 0.0], dtype=np.float32)
      velocity_xyz=np.array([0.2, 0.0, 0.0], dtype=np.float32)
      yaw=0.1
      covariance=np.eye(4, dtype=np.float32)
    """

    timestamp: float
    position_xyz: np.ndarray  # (3,) in track coord_frame
    velocity_xyz: Optional[np.ndarray]
    yaw: Optional[float]
    covariance: Optional[np.ndarray]  # (K,K) optional (Kalman)


@dataclass()
class Track:
    """
    Track with history of states.
    - track_id: unique identifier
    - class_id: optional class index
    - states: list of TrackState (history or current-only)
    - confidence: optional track confidence

    Example:
      track_id=7
      class_id=2
      states=[TrackState(...)]
      confidence=0.83
    """

    meta: OverlayMeta
    track_id: int
    class_id: Optional[int]
    states: List[TrackState]
    confidence: Optional[float]
