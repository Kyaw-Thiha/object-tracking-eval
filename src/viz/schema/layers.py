"""
Layer schema definitions.

Order of contents:
- RasterLayer
- PointLayer
- Box2DLayer
- Box3DLayer
- LineLayer
- TextLayer
- TrackLayer
"""

from dataclasses import dataclass
from typing import Literal
import numpy as np

from .base_layer import Layer


@dataclass
class RasterLayer(Layer):
    """
    Raster image or grid layer.
    - data: HxW or HxWxC array
    - axes/bins: optional axis semantics for grids (e.g., range/azimuth)
    - extent: (xmin, xmax, ymin, ymax) for plotting
    - grid_name: optional label (e.g., "RA", "RD", "RAD")
    - display: "pixel" for imshow, "polar" for cone-style or polar plots
    """

    data: np.ndarray  # (H,W or H,W,C)
    axes: tuple[str, str] | None  # (e.g., ("range","azimuth"))
    bins: dict[str, np.ndarray] | None  # (bin centers per axis)
    extent: tuple[float, float, float, float] | None
    grid_name: str | None  # e.g., "RA", "RD", "RAD"
    display: Literal["pixel", "polar"]


@dataclass
class PointLayer(Layer):
    """
    Point cloud or scatter layer.
    - xyz: (N,3) or (N,2) points
    - value: optional scalar values for colormaps
    - color: optional RGB per-point or global color
    - value_key/units: semantic label for the value field
    """

    xyz: np.ndarray  # (N,3 or N,2)
    value: np.ndarray | None  # (N,) for colormap
    color: np.ndarray | None  # (N,3 or 3,)
    value_key: str | None  # "doppler", "rcs", "snr"
    units: str | None


@dataclass
class Box2DLayer(Layer):
    """
    2D bounding boxes.
    - xyxy: (N,4) array of [x1,y1,x2,y2]
    - labels: optional text labels
    - class_ids: optional class ids aligned with boxes
    """

    xyxy: np.ndarray  # (N,4)
    labels: list[str] | None
    class_ids: np.ndarray | None


@dataclass
class Box3DLayer(Layer):
    """
    3D bounding boxes with yaw.
    - centers: (N,3)
    - sizes_lwh: (N,3)
    - yaws: (N,)
    - labels/class_ids: optional annotations
    """

    centers: np.ndarray  # (N,3)
    sizes_lwh: np.ndarray  # (N,3)
    yaws: np.ndarray  # (N,)
    labels: list[str] | None
    class_ids: np.ndarray | None


@dataclass
class LineLayer(Layer):
    """
    Line segments.
    - segments: (M,2,3) or (M,2,2)
    """

    segments: np.ndarray  # (M,2,3 or M,2,2)


@dataclass
class TextLayer(Layer):
    """
    Text labels in 2D or 3D space.
    - xy: (N,2) or (N,3)
    - texts: list of strings aligned with xy
    """

    xy: np.ndarray  # (N,2 or N,3)
    texts: list[str]


@dataclass
class TrackLayer(Layer):
    """
    Track states and optional history.
    - track_ids: (N,)
    - positions_xyz: (N,3)
    - velocities_xyz/yaws/covariances: optional kinematic data
    - labels: optional pre-formatted strings (id, age, velocity)
    - history: list of (T_i,3) arrays for per-track trails
    - source_key: explicit source ("gt" or "pred:<run_id>")
    """

    track_ids: np.ndarray  # (N,)
    positions_xyz: np.ndarray  # (N,3)
    velocities_xyz: np.ndarray | None
    yaws: np.ndarray | None
    covariances: np.ndarray | None  # (N,K,K)
    labels: list[str] | None
    history: list[np.ndarray] | None  # Each element is (T_i,3) for that track's trail
    source_key: str
    history_len: int | None
    velocity_units: str | None
