"""
Lidar schema definitions.

Order of contents:
- LidarMeta
- LidarPointCloud
- LidarSensorFrame
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass()
class LidarMeta:
    """
    Lidar metadata for a single frame.
    - frame: coordinate frame name (e.g., "sensor:lidar_top")
    - sensor_pose_in_ego: (4, 4) SE(3) transform, sensor -> ego

    Example:
      frame="sensor:lidar_top"
      sensor_pose_in_ego=np.eye(4, dtype=np.float32)
    """
    frame: str
    sensor_pose_in_ego: np.ndarray  # (4x4 SE(3) transform)


@dataclass()
class LidarPointCloud:
    """
    Lidar point cloud with optional per-point features.
    - xyz: (N, 3) points
    - features: per-point arrays aligned with xyz (N,)
    - frame: "sensor" or "ego"

    Example:
      xyz=np.zeros((2048, 3), dtype=np.float32)
      features={"intensity": np.zeros(2048, dtype=np.float32)}
      frame="sensor"
    """
    xyz: np.ndarray  # (N,3)
    features: dict[str, np.ndarray]
    frame: str


@dataclass()
class LidarSensorFrame:
    """
    Single lidar capture with metadata and point cloud.
    - point_cloud: optional sparse or dense points
    """
    sensor_id: str
    meta: LidarMeta
    point_cloud: Optional[LidarPointCloud]
