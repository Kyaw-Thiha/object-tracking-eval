from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass()
class ImageMeta:
    """
      Camera metadata for a single image.
    - spectral: "rgb" / "ir" / "uv"
    - frame: coordinate frame name (e.g., "sensor:cam_front")
    - intrinsics: (3, 3) camera matrix
    - sensor_pose_in_ego: (4, 4) SE(3) transform, sensor -> ego
    - distortion: 1D coeffs (k1,k2,p1,p2,k3,...) or None
    - exposure: exposure time in ms, if available
    - gain: sensor gain, if available

    Example:
      spectral="rgb"
      frame="sensor:cam_front"
      intrinsics=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
      distortion=np.array([k1, k2, p1, p2, k3], dtype=np.float32)
      sensor_pose_in_ego=np.eye(4, dtype=np.float32)
      exposure=10.0
      gain=1.5
    """

    spectral: str
    frame: str
    intrinsics: np.ndarray  # (3x3 camera matrix, float32)
    sensor_pose_in_ego: np.ndarray  # (4x4 SE(3) transform)
    # distortion: Optional[np.ndarray]  # (1D coeffs like k1,k2,p1,p2,k3,...)
    # exposure: Optional[float]
    # gain: Optional[float]


@dataclass()
class ImageSensorFrame:
    """
    Single image capture with metadata and optional validity mask.
    - image: HxW or HxWxC array
    - mask: optional boolean/uint8 mask for valid pixels
    - meta: image meta data like intrinsics, poses and encoding details

    Example:
      sensor_id="cam_front"
      image=np.zeros((720, 1280, 3), dtype=np.uint8)
      mask=np.ones((720, 1280), dtype=np.uint8)
    """

    sensor_id: str
    image: np.ndarray  # HxW or HxWxC
    meta: ImageMeta  # intrinsics + poses + encoding details
    mask: Optional[np.ndarray] = None  # optional validity mask
