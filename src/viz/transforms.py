from __future__ import annotations

import numpy as np


def transform_points(xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 SE(3) transform to (N,3) points.
    - xyz: (N,3)
    - T: (4,4) homogeneous transform
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be (N,3)")
    if T.shape != (4, 4):
        raise ValueError("T must be (4,4)")

    ones = np.ones((xyz.shape[0], 1), dtype=xyz.dtype)
    homo = np.concatenate([xyz, ones], axis=1)
    out = (T @ homo.T).T
    return out[:, :3]


def invert_se3(T: np.ndarray) -> np.ndarray:
    """
    Invert a 4x4 SE(3) transform.
    """
    if T.shape != (4, 4):
        raise ValueError("T must be (4,4)")

    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t

    out = np.eye(4, dtype=T.dtype)
    out[:3, :3] = R_inv
    out[:3, 3] = t_inv
    return out


def compose_se3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compose two 4x4 SE(3) transforms.
    Returns a @ b.
    """
    if a.shape != (4, 4) or b.shape != (4, 4):
        raise ValueError("a and b must be (4,4)")
    return a @ b


def transform_boxes3d(
    centers: np.ndarray,
    sizes_lwh: np.ndarray,
    yaws: np.ndarray | None,
    T: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Transform 3D boxes with a 4x4 SE(3) transform.
    - centers: (N,3)
    - sizes_lwh: (N,3)
    - yaws: (N,) or None (assumes z-up yaw)
    - T: (4,4) transform
    """
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers must be (N,3)")
    if sizes_lwh.ndim != 2 or sizes_lwh.shape[1] != 3:
        raise ValueError("sizes_lwh must be (N,3)")
    if T.shape != (4, 4):
        raise ValueError("T must be (4,4)")

    centers_out = transform_points(centers, T)
    sizes_out = sizes_lwh.copy()

    if yaws is None:
        return centers_out, sizes_out, None
    if yaws.ndim != 1:
        raise ValueError("yaws must be (N,)")

    yaw_vecs = np.stack([np.cos(yaws), np.sin(yaws), np.zeros_like(yaws)], axis=1)
    rot = T[:3, :3]
    yaw_vecs_out = (rot @ yaw_vecs.T).T
    yaws_out = np.arctan2(yaw_vecs_out[:, 1], yaw_vecs_out[:, 0])
    return centers_out, sizes_out, yaws_out


def project_points_to_image(
    xyz: np.ndarray,
    intrinsics: np.ndarray,
    image_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points in the camera frame to image pixels.
    - xyz: (N,3) points in the camera frame
    - intrinsics: (3,3) camera matrix
    - image_shape: optional (H,W) for bounds checking
    Returns:
      - uv: (N,2) projected pixel coordinates
      - mask: (N,) boolean mask for valid points
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be (N,3)")
    if intrinsics.shape != (3, 3):
        raise ValueError("intrinsics must be (3,3)")

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    valid = z > 0

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    uv = np.stack([u, v], axis=1)

    if image_shape is not None:
        h, w = image_shape
        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        valid = valid & in_bounds

    return uv, valid
