import numpy as np
from pyquaternion import Quaternion


def se3_from_quaternion(rotation: list[float], translation: list[float], dtype=np.float32) -> np.ndarray:
    """
    Build a 4x4 SE(3) matrix from a quaternion and translation.
    `rotation` should be [w, x, y, z] as in NuScenes.
    """
    quaternion = Quaternion(rotation)
    mat = np.eye(4, dtype=dtype)
    mat[:3, :3] = quaternion.rotation_matrix.astype(dtype)  # Select the top-left 3x3 block
    mat[:3, 3] = np.array(translation, dtype=dtype)  # Select the first 3 rows of last column
    return mat


def yaw_from_quaternion(rotation: list[float]) -> float:
    """rotation is [w, x, y, z] (NuScenes format)."""
    q = Quaternion(rotation)
    # yaw around +z
    return np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
