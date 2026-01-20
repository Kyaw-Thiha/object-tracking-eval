from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass()
class RadarMeta:
    pass


@dataclass()
class GridRadar:
    tensor: np.ndarray  # 2D or 3D
    axes: tuple[str, ...]  # ("range","azimuth") / ("range","doppler") / ("range","azimuth","doppler")
    bins: dict[str, np.ndarray]  # bin centers for each axis
    units: dict[str, str]  # meters, radians, m/s
    layouts: str  # explicit ordering to avoid bugs


@dataclass()
class PointCloud:
    xyz: np.ndarray  # (N,3)
    features: dict[str, np.ndarray]  # doppler, rcs, snr, time_offset, etc
    frame: str  # sensor or ego


@dataclass()
class RadarSensorFrame:
    sensor_id: str
    meta: RadarMeta
    grids: dict[str, GridRadar]
    point_cloud: Optional[PointCloud]
