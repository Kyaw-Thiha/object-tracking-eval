from __future__ import annotations

from collections import defaultdict
from typing import Any

import cv2
import numpy as np

from nuscenes.nuscenes import RadarPointCloud

from .nuscenes import NuScenesAdapter
from ..utils.transforms import se3_from_quaternion


class NuScenesRCAdapter(NuScenesAdapter):
    """NuScenes adapter extension for RCBEVDet temporal radar-camera inputs."""

    def __init__(
        self,
        dataset_name: str = "NuScenes",
        dataset_path: str = "data/nuScenes",
        synthesize_radar_grids: bool = False,
        radar_range_bins: np.ndarray | None = None,
        radar_azimuth_bins: np.ndarray | None = None,
        radar_doppler_bins: np.ndarray | None = None,
        radar_grid_value: str = "count",
    ) -> None:
        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            synthesize_radar_grids=synthesize_radar_grids,
            radar_range_bins=radar_range_bins,
            radar_azimuth_bins=radar_azimuth_bins,
            radar_doppler_bins=radar_doppler_bins,
            radar_grid_value=radar_grid_value,
        )
        self._frame_lookup: dict[tuple[str, int], dict[str, Any]] = {}
        self._scene_frames: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for info in self.frames:
            meta = info.get("meta")
            sequence_id = meta.sequence_id if meta is not None else ""
            key = (sequence_id, int(info["frame_id"]))
            self._frame_lookup[key] = info
            self._scene_frames[sequence_id].append(info)

    def index_frames(self) -> list[dict[str, Any]]:
        """Build frame index and keep sample/scene tokens for temporal lookup."""
        frames: list[dict[str, Any]] = []
        for scene in self.nusc.scene:
            token = scene["first_sample_token"]
            frame_idx = 0
            while token:
                sample = self.nusc.get("sample", token)
                frames.append(
                    {
                        "sample_token": token,
                        "scene_token": scene["token"],
                        "timestamp": sample["timestamp"] * 1e-6,
                        "frame_id": frame_idx,
                        "meta": self._build_frame_meta(scene_name=scene["name"]),
                    }
                )
                token = sample["next"]
                frame_idx += 1
        return frames

    @staticmethod
    def _build_frame_meta(scene_name: str):
        from ..schema.frame import FrameMeta

        return FrameMeta(
            sequence_id=scene_name,
            dataset="nuscenes",
            scene=None,
            split=None,
            weather=None,
        )

    def get_frame_info(self, sequence_id: str, frame_id: int) -> dict[str, Any]:
        key = (sequence_id, int(frame_id))
        if key not in self._frame_lookup:
            raise KeyError(f"No frame info for sequence={sequence_id}, frame_id={frame_id}")
        return self._frame_lookup[key]

    def get_scene_frame_infos(self, sequence_id: str) -> list[dict[str, Any]]:
        return self._scene_frames.get(sequence_id, [])

    def get_camera_sample_data(
        self,
        sample_token: str,
        camera_ids: list[str],
        load_images: bool = True,
    ) -> dict[str, dict[str, Any]]:
        sample = self.nusc.get("sample", sample_token)
        camera_data: dict[str, dict[str, Any]] = {}
        for camera_id in camera_ids:
            sample_data_token = sample["data"].get(camera_id)
            if sample_data_token is None:
                continue

            sample_data = self.nusc.get("sample_data", sample_data_token)
            calibrated_sensor = self.nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
            ego = self.nusc.get("ego_pose", sample_data["ego_pose_token"])

            image = None
            if load_images:
                image_path = self.nusc.get_sample_data_path(sample_data_token)
                image = cv2.imread(image_path)

            camera_data[camera_id] = {
                "sample_data_token": sample_data_token,
                "timestamp": sample_data["timestamp"] * 1e-6,
                "image": image,
                "intrinsics": np.array(calibrated_sensor["camera_intrinsic"], dtype=np.float32),
                "sensor_pose_in_ego": se3_from_quaternion(
                    calibrated_sensor["rotation"], calibrated_sensor["translation"]
                ),
                "ego_pose_in_world": se3_from_quaternion(ego["rotation"], ego["translation"]),
            }

        return camera_data

    def get_radar_sweeps(
        self,
        sample_token: str,
        radar_ids: list[str],
        num_sweeps: int = 9,
    ) -> dict[str, list[dict[str, Any]]]:
        sample = self.nusc.get("sample", sample_token)
        current_timestamp = sample["timestamp"] * 1e-6

        sweeps_by_radar: dict[str, list[dict[str, Any]]] = {}
        for radar_id in radar_ids:
            radar_sweeps: list[dict[str, Any]] = []
            sample_data_token = sample["data"].get(radar_id)
            while sample_data_token and len(radar_sweeps) < num_sweeps:
                sample_data = self.nusc.get("sample_data", sample_data_token)
                calibrated_sensor = self.nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
                ego = self.nusc.get("ego_pose", sample_data["ego_pose_token"])

                point_cloud_path = self.nusc.get_sample_data_path(sample_data_token)
                point_cloud = RadarPointCloud.from_file(point_cloud_path)
                xyz = point_cloud.points[:3, :].T.astype(np.float32)

                # Keep both raw and compensated velocities when available.
                vx = point_cloud.points[6, :].astype(np.float32)
                vy = point_cloud.points[7, :].astype(np.float32)
                if point_cloud.points.shape[0] > 9:
                    vx_comp = point_cloud.points[8, :].astype(np.float32)
                    vy_comp = point_cloud.points[9, :].astype(np.float32)
                else:
                    vx_comp = vx
                    vy_comp = vy
                rcs = point_cloud.points[5, :].astype(np.float32)

                radar_sweeps.append(
                    {
                        "sample_data_token": sample_data_token,
                        "timestamp": sample_data["timestamp"] * 1e-6,
                        "time_lag": current_timestamp - sample_data["timestamp"] * 1e-6,
                        "sensor_pose_in_ego": se3_from_quaternion(
                            calibrated_sensor["rotation"], calibrated_sensor["translation"]
                        ),
                        "ego_pose_in_world": se3_from_quaternion(ego["rotation"], ego["translation"]),
                        "xyz": xyz,
                        "features": {
                            "vx": vx,
                            "vy": vy,
                            "vx_comp": vx_comp,
                            "vy_comp": vy_comp,
                            "rcs": rcs,
                        },
                    }
                )

                sample_data_token = sample_data.get("prev")

            sweeps_by_radar[radar_id] = radar_sweeps

        return sweeps_by_radar
