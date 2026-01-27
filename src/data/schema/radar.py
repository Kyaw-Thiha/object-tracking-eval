from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass()
class RadarMeta:
    """
    Radar metadata for a single frame.
    - frame: coordinate frame name (e.g., "sensor:lidar_top")
    - sensor_pose_in_ego: (4, 4) SE(3) transform, sensor -> ego

    Example:
      frame="sensor:lidar_top"
      sensor_pose_in_ego=np.eye(4, dtype=np.float32)
      ego_pose_in_world: (4, 4) SE(3) transform, ego -> world (optional)
      ego_velocity_in_world=np.array([vx, vy, vz], dtype=np.float32)
    """

    frame: str
    sensor_pose_in_ego: np.ndarray  # (4x4 SE(3) transform), sensor -> ego
    ego_pose_in_world: Optional[np.ndarray] = None  # (4x4 SE(3) transform), ego -> world
    ego_velocity_in_world: Optional[np.ndarray] = None  # (3,) velocity in world (m/s)


@dataclass()
class GridRadar:
    """
    Radar grid product with axis semantics.
    - tensor: 2D or 3D grid (e.g., range-azimuth, range-doppler, range-azimuth-doppler)
    - axes/layouts: axis names and explicit order that must match tensor dimensions
    - bins/units: per-axis bin centers and physical units (keys must match axes)

    Example:
      tensor=np.zeros((len(bins["range"]), len(bins["azimuth"])))
      axes=("range", "azimuth")
      layouts="R,A"
      bins={"range": np.arange(0, 80, 0.2), "azimuth": np.linspace(-1.2, 1.2, 128)}
      units={"range": "m", "azimuth": "rad"}
    """

    tensor: np.ndarray  # 2D or 3D
    axes: tuple[str, ...]  # ("range","azimuth") / ("range","doppler") / ("range","azimuth","doppler")
    layouts: str  # "R,A,D" or "R,A" or "R,D": explicit ordering to avoid bugs
    bins: dict[str, np.ndarray]  # bin centers for each axis
    units: dict[str, str]  # range: meters, azimuth: radians, doppler: m/s


@dataclass()
class PointCloud:
    """
    Radar point detections in a named coordinate frame.
    - xyz: (N, 3) points
    - features: per-point arrays aligned with xyz (N,)
    - frame: "sensor" or "ego"

    Example:
      xyz=np.zeros((128, 3))
      features={"doppler": np.zeros(128), "rcs": np.zeros(128), "snr": np.zeros(128)}
      frame="sensor"
    """

    xyz: np.ndarray  # (N,3)
    features: dict[str, np.ndarray]  # doppler, rcs, snr, time_offset, etc
    frame: str  # sensor or ego


@dataclass()
class RadarSensorFrame:
    """
    Single radar capture with metadata and products.
    - grids: named grid products (RA/RD/RAD)
    - point_cloud: optional sparse detections
    """

    sensor_id: str
    meta: RadarMeta
    grids: Optional[dict[str, GridRadar]] = None
    point_cloud: Optional[PointCloud] = None
