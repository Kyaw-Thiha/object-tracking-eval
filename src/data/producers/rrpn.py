"""RRPN Producer: generates region proposals from radar + camera."""
import numpy as np
import torch
import cv2 as cv
from typing import Any, cast
from numpy.typing import NDArray

from ..schema.frame import Frame
from ..schema.image import ImageSensorFrame
from ..schema.overlay import Box3D
from ..schema.radar import RadarSensorFrame
from .base import BaseProducer


def project_radar_to_image(
    xyz: NDArray[np.floating[Any]],
    K: NDArray[np.floating[Any]],
    T: NDArray[np.floating[Any]],
    img_shape: tuple[int, int],
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], NDArray[np.bool_]]:
    """
    Project 3D radar points into image pixel coordinates.

    Args:
      xyz: Radar points in radar/sensor frame with shape (N, 3).
      K: Camera intrinsic matrix with shape (3, 3).
      T: Rigid transform from radar frame to camera frame with shape (4, 4).
      img_shape: Image shape as (H, W).

    Returns:
      uv_valid: Pixel coordinates (u, v) for valid projected points, shape (M, 2).
      depths_valid: Camera-frame depths for valid points, shape (M,).
      valid_mask: Boolean mask over original N points indicating which points are valid.
    """
    # Convert xyz to homogeneous coordinates so we can apply a single 4x4 transform.
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1))])

    # Transform points from radar frame into camera frame.
    cam_pts = (T @ xyz_h.T).T[:, :3]

    # Keep only points with physically meaningful forward depth.
    depths = cam_pts[:, 2]
    valid = (depths > 1.0) & (depths < 100.0)

    # Project camera-frame 3D points onto the image plane using intrinsics.
    uv_h = (K @ cam_pts.T).T
    uv = uv_h[:, :2] / (depths[:, None] + 1e-6)

    # Keep only projections that land inside image bounds.
    h, w = img_shape
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)

    # Return filtered coordinates/depths plus the original-length validity mask.
    return uv[valid], depths[valid], valid


def generate_anchors(
    pois: NDArray[np.floating[Any]],
    scales: NDArray[np.floating[Any]],
    sizes: tuple[int, ...] = (32, 64, 128, 256),
    ratios: tuple[float, ...] = (0.5, 1.0, 2.0),
) -> NDArray[np.float32]:
    """
    Generate axis-aligned anchor boxes around radar-driven points of interest (POIs).

    For each POI, multiple base sizes and aspect ratios are combined with a per-point
    scale factor. For each resulting box shape, several center offsets are generated to
    increase local proposal coverage.

    Args:
      pois: Pixel POIs with shape (N, 2), where each row is (u, v).
      scales: Per-POI scale multipliers with shape (N,).
      sizes: Base anchor sizes in pixels.
      ratios: Width/height aspect ratios for each base size.

    Returns:
      Anchor boxes as (M, 4) float32 array in [x1, y1, x2, y2] format.
      Returns shape (0, 4) when no anchors are generated.
    """
    anchors = []
    for (u, v), s in zip(pois, scales):
        for size in sizes:
            for r in ratios:
                w, h = size * np.sqrt(r) * s, size / np.sqrt(r) * s
                # Center + 3 offset alignments
                for dx, dy in [(0, 0), (w/2, 0), (-w/2, 0), (0, h/2)]:
                    x1, y1 = u - w/2 + dx, v - h/2 + dy
                    x2, y2 = x1 + w, y1 + h
                    anchors.append([x1, y1, x2, y2])
    return np.array(anchors, dtype=np.float32) if anchors else np.zeros((0, 4), dtype=np.float32)


class RRPNProducer(BaseProducer):
    """Producer for RRPN: generates proposals from radar detections."""

    def __init__(self, adapter, input_size=(800, 1600), camera_id="CAM_FRONT",
                 radar_ids=None, alpha=20.0, beta=0.5):
        super().__init__(adapter)
        self.input_size = input_size
        self.camera_id = camera_id
        self.radar_ids = radar_ids or ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT"]
        self.alpha, self.beta = alpha, beta  # Distance compensation: S = α/d + β

    @staticmethod
    def _find_ego_pose_in_world(frame: Frame) -> NDArray[np.floating[Any]] | None:
        for sensor in frame.sensors.values():
            meta = getattr(sensor.data, "meta", None)
            pose = getattr(meta, "ego_pose_in_world", None) if meta is not None else None
            if pose is not None:
                return pose
        return None

    @staticmethod
    def _box3d_corners_world(box: Box3D) -> NDArray[np.float32]:
        l, w, h = box.size_lwh.astype(np.float32)
        x = l / 2.0
        y = w / 2.0
        z = h / 2.0
        # 8 corners centered at origin, then yaw-rotated and translated.
        local = np.array(
            [
                [x, y, z],
                [x, -y, z],
                [-x, -y, z],
                [-x, y, z],
                [x, y, -z],
                [x, -y, -z],
                [-x, -y, -z],
                [-x, y, -z],
            ],
            dtype=np.float32,
        )
        c = np.float32(np.cos(box.yaw))
        s = np.float32(np.sin(box.yaw))
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        return (R @ local.T).T + box.center_xyz.astype(np.float32)

    @staticmethod
    def _project_box3d_to_image(
        box: Box3D,
        K: NDArray[np.floating[Any]],
        cam_pose_in_ego: NDArray[np.floating[Any]],
        ego_pose_in_world: NDArray[np.floating[Any]],
        img_shape: tuple[int, int],
    ) -> NDArray[np.float32] | None:
        corners_world = RRPNProducer._box3d_corners_world(box)
        corners_h = np.hstack([corners_world, np.ones((corners_world.shape[0], 1), dtype=np.float32)])

        # world -> ego -> camera
        world_to_ego = np.linalg.inv(ego_pose_in_world)
        ego_to_cam = np.linalg.inv(cam_pose_in_ego)
        corners_ego = (world_to_ego @ corners_h.T).T
        corners_cam = (ego_to_cam @ corners_ego.T).T[:, :3]

        z = corners_cam[:, 2]
        valid = z > 1e-3
        if not np.any(valid):
            return None

        uv_h = (K @ corners_cam[valid].T).T
        uv = uv_h[:, :2] / (corners_cam[valid, 2:3] + 1e-6)

        h, w = img_shape
        x1 = float(np.clip(np.min(uv[:, 0]), 0.0, w - 1.0))
        y1 = float(np.clip(np.min(uv[:, 1]), 0.0, h - 1.0))
        x2 = float(np.clip(np.max(uv[:, 0]), 0.0, w - 1.0))
        y2 = float(np.clip(np.max(uv[:, 1]), 0.0, h - 1.0))
        if x2 <= x1 or y2 <= y1:
            return None
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def get_sensors(self, frame: Frame) -> dict[str, Any]:
        cam_sf = frame.sensors.get(self.camera_id)
        if not cam_sf or cam_sf.kind != "image":
            return {"image": None, "radar_pois": None}

        cam_data = cam_sf.data
        if not isinstance(cam_data, ImageSensorFrame):
            return {"image": None, "radar_pois": None}

        img = cam_data.image
        K = cam_data.meta.intrinsics
        cam_pose = cam_data.meta.sensor_pose_in_ego

        # Aggregate radar from all sensors
        all_pois, all_dists, all_vels, all_rcs = [], [], [], []
        for rid in self.radar_ids:
            radar_sf = frame.sensors.get(rid)
            if not radar_sf or radar_sf.kind != "radar":
                continue

            radar = radar_sf.data
            if not isinstance(radar, RadarSensorFrame) or radar.point_cloud is None:
                continue

            xyz = radar.point_cloud.xyz
            radar_pose = radar.meta.sensor_pose_in_ego
            T = np.linalg.inv(cam_pose) @ radar_pose

            img_h, img_w = img.shape[:2]
            uv, d, mask = project_radar_to_image(xyz, K, T, (int(img_h), int(img_w)))
            if len(uv) > 0:
                feats = radar.point_cloud.features
                all_pois.append(uv)
                all_dists.append(d)
                all_vels.append(feats.get("doppler", np.zeros(len(uv)))[mask] if "doppler" in feats else np.zeros(len(uv)))
                all_rcs.append(feats.get("rcs", np.ones(len(uv)))[mask] if "rcs" in feats else np.ones(len(uv)))

        if not all_pois:
            return {"image": img, "radar_pois": None}

        return {
            "image": img,
            "radar_pois": np.vstack(all_pois),
            "radar_dists": np.hstack(all_dists),
            "radar_vels": np.hstack(all_vels),
            "radar_rcs": np.hstack(all_rcs),
        }

    def preprocess(self, sensors: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        img = sensors["image"]
        if img is None:
            return {"image": torch.zeros((3, *self.input_size))}, {}

        # Resize + letterbox
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        ratio = min(self.input_size[0] / img.shape[0], self.input_size[1] / img.shape[1])
        new_h, new_w = int(img.shape[0] * ratio), int(img.shape[1] * ratio)
        resized = cv.resize(img_rgb, (new_w, new_h)).astype(np.float32)
        padded = np.ones((*self.input_size, 3), dtype=np.float32) * 114.0
        padded[:new_h, :new_w] = resized

        # Generate proposals from radar
        proposals = np.zeros((0, 4), dtype=np.float32)
        radar_meta = {}

        if sensors["radar_pois"] is not None:
            pois = sensors["radar_pois"] * ratio  # Scale POIs
            dists = sensors["radar_dists"]
            scales = self.alpha / (dists + 1e-3) + self.beta
            scales = np.clip(scales, 0.1, 10.0)

            proposals = generate_anchors(pois, scales)
            # Clip to image
            proposals[:, [0, 2]] = np.clip(proposals[:, [0, 2]], 0, self.input_size[1])
            proposals[:, [1, 3]] = np.clip(proposals[:, [1, 3]], 0, self.input_size[0])

            # Filter small boxes
            w, h = proposals[:, 2] - proposals[:, 0], proposals[:, 3] - proposals[:, 1]
            valid = (w > 10) & (h > 10)
            proposals = proposals[valid]

            radar_meta = {
                "pois": pois,
                "range": dists,
                "velocity": sensors["radar_vels"],
                "rcs": sensors["radar_rcs"],
            }

        return {
            "image": torch.from_numpy(padded).permute(2, 0, 1),
        }, {
            "proposals": torch.from_numpy(proposals),
            "radar_meta": radar_meta,
            "scale_factor": ratio,
            "ori_shape": img.shape,
        }

    def build_target(self, frame: Frame, meta: dict[str, Any]) -> dict[str, Any]:
        # Standard targets + RRPN context
        boxes, labels = [], []
        if frame.overlays and "gt" in frame.overlays.boxes2D:
            for box in frame.overlays.boxes2D["gt"]:
                x1, y1, x2, y2 = box.xyxy.tolist()
                boxes.append([x1, y1, x2, y2])
                labels.append(int(box.class_id))
        elif frame.overlays and "gt" in frame.overlays.boxes3D:
            cam_sf = frame.sensors.get(self.camera_id)
            if cam_sf and cam_sf.kind == "image" and isinstance(cam_sf.data, ImageSensorFrame):
                cam_data = cam_sf.data
                K = cam_data.meta.intrinsics
                cam_pose = cam_data.meta.sensor_pose_in_ego
                ego_pose_world = self._find_ego_pose_in_world(frame)
                if ego_pose_world is not None:
                    img_h, img_w = cam_data.image.shape[:2]
                    for box3d in frame.overlays.boxes3D["gt"]:
                        xyxy = self._project_box3d_to_image(box3d, K, cam_pose, ego_pose_world, (int(img_h), int(img_w)))
                        if xyxy is None:
                            continue
                        boxes.append(xyxy.tolist())
                        labels.append(int(box3d.class_id))

        # Get proposals from preprocess meta
        proposals = meta.get("proposals", torch.zeros((0, 4), dtype=torch.float32))
        scale_factor = float(meta.get("scale_factor", 1.0))
        ori_shape = cast(tuple[int, int, int], meta.get("ori_shape", (self.input_size[0], self.input_size[1], 3)))
        pad_shape = (self.input_size[0], self.input_size[1], 3)
        video_id = frame.meta.sequence_id if frame.meta and frame.meta.sequence_id else "sequence_0"
        boxes_t = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long)

        return {
            "boxes": boxes_t,
            "labels": labels_t,
            "gt_bboxes": boxes_t,
            "gt_labels": labels_t,
            "proposals": proposals,  # RRPN proposals
            "frame_id": frame.frame_id,
            "video_id": video_id,
            "img_metas": {
                "scale_factor": scale_factor,
                "ori_shape": ori_shape,
                "img_shape": pad_shape,
                "pad_shape": pad_shape,
            },
            "radar_meta": meta.get("radar_meta", {}),
        }
