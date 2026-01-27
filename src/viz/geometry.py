from __future__ import annotations

import numpy as np

from .transforms import project_points_to_image, transform_points
from ..data.schema.frame import Frame


def box3d_corners(center_xyz: np.ndarray, size_lwh: np.ndarray, yaw: float) -> np.ndarray:
    """
    Return 8 corners for a 3D box in its coord frame, yaw around +z.
    center_xyz: (3,)
    size_lwh: (3,) -> (length, width, height)
    """
    l, w, h = size_lwh.tolist()
    hx = l / 2
    hy = w / 2
    hz = h / 2
    corners = np.array(
        [
            [hx, hy, -hz],
            [hx, -hy, -hz],
            [-hx, -hy, -hz],
            [-hx, hy, -hz],
            [hx, hy, hz],
            [hx, -hy, hz],
            [-hx, -hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    return (R @ corners.T).T + center_xyz[None, :]


def box3d_bev_corners(center_xyz: np.ndarray, size_lwh: np.ndarray, yaw: float) -> np.ndarray:
    """
    Return 2D BEV corners (closed polyline) in xy plane.
    """
    l, w, _ = size_lwh.tolist()
    corners = np.array(
        [
            [l / 2, w / 2],
            [l / 2, -w / 2],
            [-l / 2, -w / 2],
            [-l / 2, w / 2],
            [l / 2, w / 2],
        ],
        dtype=np.float32,
    )
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return (R @ corners.T).T + center_xyz[:2][None, :]


def box3d_to_box2d(
    center_xyz: np.ndarray,
    size_lwh: np.ndarray,
    yaw: float,
    T_cam_world: np.ndarray,
    intrinsics: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    """
    Project 3D box to 2D axis-aligned pixel box.
    """
    corners = box3d_corners(center_xyz, size_lwh, yaw)
    corners_cam = transform_points(corners, T_cam_world)
    uv, valid = project_points_to_image(corners_cam, intrinsics, image_shape)
    if not np.any(valid):
        return None
    uv = uv[valid]
    x1, y1 = np.min(uv, axis=0).tolist()
    x2, y2 = np.max(uv, axis=0).tolist()
    return (x1, y1, x2, y2)


def ego_pose_in_world_from_frame(frame: Frame) -> np.ndarray | None:
    """
    Return the first available ego_pose_in_world from any sensor meta in a frame.
    """
    for sensor in frame.sensors.values():
        meta = getattr(sensor.data, "meta", None)
        if meta is None:
            continue
        pose = getattr(meta, "ego_pose_in_world", None)
        if pose is not None:
            return pose
    return None
