from __future__ import annotations

from typing import Any

import cv2 as cv
import numpy as np
import torch

from ..adapters.nuscenes_rc import NuScenesRCAdapter
from ..schema.frame import Frame
from ..schema.overlay import Box3D
from .base import BaseProducer


NUSCENES_CAMERA_IDS = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

NUSCENES_RADAR_IDS = [
    "RADAR_FRONT",
    "RADAR_FRONT_LEFT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_BACK_RIGHT",
]


class RCBEVDetProducer(BaseProducer):
    """Prepare multi-view camera + multi-sweep radar inputs for RCBEVDet."""

    def __init__(
        self,
        adapter: NuScenesRCAdapter,
        camera_ids: list[str] | None = None,
        radar_ids: list[str] | None = None,
        num_sweeps: int = 9,
        input_size: tuple[int, int] = (256, 704),
    ) -> None:
        super().__init__(adapter)
        self.adapter = adapter
        self.camera_ids = camera_ids or NUSCENES_CAMERA_IDS
        self.radar_ids = radar_ids or NUSCENES_RADAR_IDS
        self.num_sweeps = int(num_sweeps)
        self.input_size = input_size  # (H, W)

        # Common BEVDet-style normalization stats.
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def get_sensors(self, frame: Frame) -> dict[str, Any]:
        if frame.meta is None or frame.meta.sequence_id is None:
            raise ValueError("RCBEVDetProducer requires frame.meta.sequence_id")

        frame_info = self.adapter.get_frame_info(frame.meta.sequence_id, frame.frame_id)
        sample_token = frame_info["sample_token"]

        cameras = self.adapter.get_camera_sample_data(sample_token, self.camera_ids, load_images=False)
        for camera_id in self.camera_ids:
            sensor_frame = frame.sensors.get(camera_id)
            if sensor_frame is None or sensor_frame.kind != "image":
                continue
            image_data = getattr(sensor_frame.data, "image", None)
            if camera_id in cameras:
                cameras[camera_id]["image"] = image_data
        radar_sweeps = self.adapter.get_radar_sweeps(
            sample_token,
            radar_ids=self.radar_ids,
            num_sweeps=self.num_sweeps,
        )

        if not cameras:
            raise RuntimeError(f"No camera data loaded for sample_token={sample_token}")

        return {
            "sample_token": sample_token,
            "cameras": cameras,
            "radar_sweeps": radar_sweeps,
        }

    @staticmethod
    def _transform_points(
        xyz: np.ndarray,
        sensor_pose_in_ego: np.ndarray,
        ego_pose_in_world: np.ndarray,
        world_to_ego_current: np.ndarray,
    ) -> np.ndarray:
        n = xyz.shape[0]
        xyz_h = np.hstack([xyz, np.ones((n, 1), dtype=np.float32)])
        xyz_world = (ego_pose_in_world @ sensor_pose_in_ego @ xyz_h.T).T
        xyz_ego_current = (world_to_ego_current @ xyz_world.T).T[:, :3]
        return xyz_ego_current.astype(np.float32)

    @staticmethod
    def _transform_velocity(
        vx: np.ndarray,
        vy: np.ndarray,
        sensor_pose_in_ego: np.ndarray,
        ego_pose_in_world: np.ndarray,
        world_to_ego_current: np.ndarray,
    ) -> np.ndarray:
        vel_sensor = np.stack([vx, vy, np.zeros_like(vx)], axis=1)
        rot = (
            world_to_ego_current[:3, :3]
            @ ego_pose_in_world[:3, :3]
            @ sensor_pose_in_ego[:3, :3]
        )
        vel_ego_current = (rot @ vel_sensor.T).T
        return vel_ego_current[:, :2].astype(np.float32)

    def preprocess(self, sensors: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        cameras = sensors["cameras"]

        # Use CAM_FRONT ego pose as reference current ego frame.
        if "CAM_FRONT" not in cameras:
            raise RuntimeError("CAM_FRONT is required for RCBEVDet temporal alignment")
        ego_pose_current = cameras["CAM_FRONT"]["ego_pose_in_world"].astype(np.float32)
        world_to_ego_current = np.linalg.inv(ego_pose_current).astype(np.float32)

        h_out, w_out = self.input_size

        image_tensors: list[torch.Tensor] = []
        sensor2egos: list[torch.Tensor] = []
        ego2globals: list[torch.Tensor] = []
        intrins: list[torch.Tensor] = []
        post_rots: list[torch.Tensor] = []
        post_trans: list[torch.Tensor] = []

        ori_shape_front: tuple[int, int, int] | None = None
        scale_factor_front = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        for camera_id in self.camera_ids:
            if camera_id not in cameras:
                raise RuntimeError(f"Missing camera '{camera_id}' for RCBEVDet input")
            cam = cameras[camera_id]
            image = cam["image"]
            if image is None:
                raise RuntimeError(f"Failed to load image for camera '{camera_id}'")

            ori_h, ori_w = image.shape[:2]
            if camera_id == "CAM_FRONT":
                ori_shape_front = (ori_h, ori_w, 3)

            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            resized = cv.resize(image_rgb, (w_out, h_out), interpolation=cv.INTER_LINEAR).astype(np.float32)
            resized = (resized - self.mean) / self.std
            image_tensor = torch.from_numpy(resized).permute(2, 0, 1).contiguous().float()
            image_tensors.append(image_tensor)

            sx = float(w_out) / float(ori_w)
            sy = float(h_out) / float(ori_h)
            if camera_id == "CAM_FRONT":
                scale_factor_front = np.array([sx, sy, sx, sy], dtype=np.float32)

            intr = cam["intrinsics"].astype(np.float32).copy()
            intr[0, 0] *= sx
            intr[0, 2] *= sx
            intr[1, 1] *= sy
            intr[1, 2] *= sy

            post_rot = np.eye(3, dtype=np.float32)
            post_rot[0, 0] = sx
            post_rot[1, 1] = sy
            post_tran = np.zeros((3,), dtype=np.float32)

            sensor2egos.append(torch.from_numpy(cam["sensor_pose_in_ego"].astype(np.float32)))
            ego2globals.append(torch.from_numpy(cam["ego_pose_in_world"].astype(np.float32)))
            intrins.append(torch.from_numpy(intr))
            post_rots.append(torch.from_numpy(post_rot))
            post_trans.append(torch.from_numpy(post_tran))

        # Aggregate multi-sweep radar points in current ego frame.
        radar_points_per_sensor: list[np.ndarray] = []
        radar_sweeps = sensors["radar_sweeps"]
        for radar_id in self.radar_ids:
            sweeps = radar_sweeps.get(radar_id, [])
            for sweep_idx, sweep in enumerate(sweeps):
                xyz_sensor = sweep["xyz"].astype(np.float32)
                if xyz_sensor.size == 0:
                    continue

                xyz_ego = self._transform_points(
                    xyz=xyz_sensor,
                    sensor_pose_in_ego=sweep["sensor_pose_in_ego"].astype(np.float32),
                    ego_pose_in_world=sweep["ego_pose_in_world"].astype(np.float32),
                    world_to_ego_current=world_to_ego_current,
                )

                features = sweep["features"]
                vx = features.get("vx_comp", features["vx"]).astype(np.float32)
                vy = features.get("vy_comp", features["vy"]).astype(np.float32)
                vel_ego = self._transform_velocity(
                    vx=vx,
                    vy=vy,
                    sensor_pose_in_ego=sweep["sensor_pose_in_ego"].astype(np.float32),
                    ego_pose_in_world=sweep["ego_pose_in_world"].astype(np.float32),
                    world_to_ego_current=world_to_ego_current,
                )
                rcs = features["rcs"].astype(np.float32)[:, None]
                sweep_col = np.full((xyz_ego.shape[0], 1), float(sweep_idx), dtype=np.float32)

                stacked = np.concatenate([xyz_ego, vel_ego, rcs, sweep_col], axis=1)
                radar_points_per_sensor.append(stacked)

        if radar_points_per_sensor:
            radar_points = torch.from_numpy(np.concatenate(radar_points_per_sensor, axis=0)).float()
        else:
            radar_points = torch.zeros((0, 7), dtype=torch.float32)

        images = torch.stack(image_tensors, dim=0)  # (6, 3, H, W)
        img_inputs = {
            "imgs": images,
            "sensor2egos": torch.stack(sensor2egos, dim=0),
            "ego2globals": torch.stack(ego2globals, dim=0),
            "intrins": torch.stack(intrins, dim=0),
            "post_rots": torch.stack(post_rots, dim=0),
            "post_trans": torch.stack(post_trans, dim=0),
            "bda": torch.eye(3, dtype=torch.float32),
        }

        if ori_shape_front is None:
            ori_shape_front = (h_out, w_out, 3)

        meta = {
            "sample_token": sensors["sample_token"],
            "img_inputs": img_inputs,
            "radar_points": radar_points,
            "img_metas": {
                "img_shape": (h_out, w_out, 3),
                "ori_shape": ori_shape_front,
                "pad_shape": (h_out, w_out, 3),
                "scale_factor": scale_factor_front,
                "flip": False,
                "flip_direction": None,
            },
        }

        return {"images": images}, meta

    def build_target(self, frame: Frame, meta: dict[str, Any]) -> dict[str, Any]:
        gt_bboxes: list[list[float]] = []
        gt_labels: list[int] = []

        overlays = frame.overlays
        if overlays and "gt" in overlays.boxes3D:
            for box in overlays.boxes3D["gt"]:
                if not isinstance(box, Box3D):
                    continue
                vx = 0.0
                vy = 0.0
                if box.velocity_xyz is not None:
                    vx = float(box.velocity_xyz[0])
                    vy = float(box.velocity_xyz[1])
                gt_bboxes.append(
                    [
                        float(box.center_xyz[0]),
                        float(box.center_xyz[1]),
                        float(box.center_xyz[2]),
                        float(box.size_lwh[0]),
                        float(box.size_lwh[1]),
                        float(box.size_lwh[2]),
                        float(box.yaw),
                        vx,
                        vy,
                    ]
                )
                gt_labels.append(int(box.class_id))

        gt_bboxes_3d = torch.tensor(gt_bboxes, dtype=torch.float32) if gt_bboxes else torch.zeros((0, 9), dtype=torch.float32)
        gt_labels_3d = torch.tensor(gt_labels, dtype=torch.long) if gt_labels else torch.zeros((0,), dtype=torch.long)

        return {
            "sample_token": meta["sample_token"],
            "video_id": frame.meta.sequence_id if frame.meta is not None else "",
            "frame_id": frame.frame_id,
            "img_metas": meta["img_metas"],
            "img_inputs": meta["img_inputs"],
            "radar_points": meta["radar_points"],
            "gt_bboxes_3d": gt_bboxes_3d,
            "gt_labels_3d": gt_labels_3d,
        }
