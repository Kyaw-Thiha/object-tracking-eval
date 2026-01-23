from dataclasses import dataclass
from typing import Literal
import numpy as np


@dataclass
class RasterLayer:
    data: np.ndarray  # (H,W or H,W,C)
    axes: tuple[str, str] | None  # (e.g., ("range","azimuth"))
    bins: dict[str, np.ndarray] | None  # (bin centers per axis)
    extent: tuple[float, float, float, float] | None
    grid_name: str | None  # e.g., "RA", "RD", "RAD"
    display: Literal["pixel", "polar"]


@dataclass
class PointLayer:
    xyz: np.ndarray  # (N,3 or N,2)
    value: np.ndarray | None  # (N,) for colormap
    color: np.ndarray | None  # (N,3 or 3,)
    value_key: str | None  # "doppler", "rcs", "snr"
    units: str | None


@dataclass
class Box2DLayer:
    xyxy: np.ndarray  # (N,4)
    labels: list[str] | None
    class_ids: np.ndarray | None


@dataclass
class Box3DLayer:
    centers: np.ndarray  # (N,3)
    sizes_lwh: np.ndarray  # (N,3)
    yaws: np.ndarray  # (N,)
    labels: list[str] | None
    class_ids: np.ndarray | None


@dataclass
class LineLayer:
    segments: np.ndarray  # (M,2,3 or M,2,2)


@dataclass
class TextLayer:
    xy: np.ndarray  # (N,2 or N,3)
    texts: list[str]


@dataclass
class TrackLayer:
    track_ids: np.ndarray  # (N,)
    positions_xyz: np.ndarray  # (N,3)
    velocities_xyz: np.ndarray | None
    yaws: np.ndarray | None
    covariances: np.ndarray | None  # (N,K,K)
    labels: list[str] | None
    history: list[np.ndarray] | None  # - Each element is (T_i,3) for that track’s trail
    source_key: str
    history_len: int | None
    velocity_units: str | None
