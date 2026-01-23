import cv2
import numpy as np


from ..schema.overlay import Box3D, OverlayMeta, OverlaySet
from ..schema.frame import FrameMeta, SensorFrame
from ..schema.image import ImageMeta, ImageSensorFrame
from ..schema.lidar import LidarMeta, LidarPointCloud, LidarSensorFrame
from ..schema.radar import GridRadar, PointCloud, RadarMeta, RadarSensorFrame
from .base import BaseAdapter
from ..utils.transforms import se3_from_quaternion, yaw_from_quaternion

from nuscenes.nuscenes import NuScenes, RadarPointCloud
from nuscenes.nuscenes import LidarPointCloud as NuScenesLidarPointCloud

NUSCENES_CLASS_MAP = {
    # vehicles
    "vehicle.car": 0,
    "vehicle.truck": 1,
    "vehicle.bus.rigid": 2,
    "vehicle.bus.bendy": 2,
    "vehicle.trailer": 3,
    "vehicle.construction": 4,
    # pedestrians
    "human.pedestrian.adult": 5,
    "human.pedestrian.child": 5,
    "human.pedestrian.construction_worker": 5,
    "human.pedestrian.police_officer": 5,
    # riders
    "vehicle.motorcycle": 6,
    "vehicle.bicycle": 7,
    # objects
    "movable_object.trafficcone": 8,
    "movable_object.barrier": 9,
}


class NuScenesAdapter(BaseAdapter):
    nusc: NuScenes
    class_map = NUSCENES_CLASS_MAP

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
        self.nusc = NuScenes(version="v1.0-mini", dataroot=dataset_path, verbose=True)
        self.instance_id_map = self.build_instance_map()
        self.synthesize_radar_grids = synthesize_radar_grids
        self.radar_grid_value = radar_grid_value
        self.radar_range_bins = radar_range_bins if radar_range_bins is not None else np.arange(0.0, 80.0 + 0.5, 0.5)
        self.radar_azimuth_bins = (
            radar_azimuth_bins if radar_azimuth_bins is not None else np.linspace(-1.2, 1.2, 256)
        )
        self.radar_doppler_bins = (
            radar_doppler_bins if radar_doppler_bins is not None else np.linspace(-20.0, 20.0, 161)
        )
        super().__init__(dataset_name, dataset_path)

    def build_radar_grids(self, xyz: np.ndarray, features: dict[str, np.ndarray]) -> dict[str, GridRadar]:
        x = xyz[:, 0]
        y = xyz[:, 1]
        rng = np.sqrt(x**2 + y**2)
        az = np.arctan2(y, x)

        doppler = features.get("doppler")
        rcs = features.get("rcs")

        weights = None
        if self.radar_grid_value == "rcs" and rcs is not None:
            weights = rcs

        ra, r_edges, a_edges = np.histogram2d(
            rng,
            az,
            bins=[self.radar_range_bins, self.radar_azimuth_bins],
            weights=weights,
        )
        ra_bins = {
            "range": 0.5 * (r_edges[:-1] + r_edges[1:]),
            "azimuth": 0.5 * (a_edges[:-1] + a_edges[1:]),
        }
        ra_grid = GridRadar(
            tensor=ra,
            axes=("range", "azimuth"),
            layouts="R,A",
            bins=ra_bins,
            units={"range": "m", "azimuth": "rad"},
        )

        grids: dict[str, GridRadar] = {"RA": ra_grid}

        if doppler is not None:
            rd, r_edges, d_edges = np.histogram2d(
                rng,
                doppler,
                bins=[self.radar_range_bins, self.radar_doppler_bins],
                weights=weights,
            )
            rd_bins = {
                "range": 0.5 * (r_edges[:-1] + r_edges[1:]),
                "doppler": 0.5 * (d_edges[:-1] + d_edges[1:]),
            }
            grids["RD"] = GridRadar(
                tensor=rd,
                axes=("range", "doppler"),
                layouts="R,D",
                bins=rd_bins,
                units={"range": "m", "doppler": "m/s"},
            )

        return grids

    # Mapping
    def build_instance_map(self) -> dict[str, int]:
        instance_map: dict[str, int] = {}
        next_id = 0
        for inst in self.nusc.instance:
            token = inst["token"]
            instance_map[token] = next_id
            next_id += 1
        return instance_map

    # Getters
    def get_sequence_ids(self) -> list[str]:
        """Return available sequence ids (scene names)."""
        return [scene["name"] for scene in self.nusc.scene]

    def get_sensor_ids(self) -> list[str]:
        """Return available sensor channel names."""
        return [sensor["channel"] for sensor in self.nusc.sensor]

    # Helpers
    def index_frames(self) -> list[dict]:
        """Build and return a list of per-frame metadata dicts used by __getitem__."""
        frames = []
        for scene in self.nusc.scene:
            token = scene["first_sample_token"]
            frame_idx = 0

            while token:
                sample = self.nusc.get("sample", token)
                frames.append(
                    {
                        "sample_token": token,
                        "timestamp": sample["timestamp"] * 1e-6,
                        "frame_id": frame_idx,
                        "meta": FrameMeta(
                            sequence_id=scene["name"],
                            dataset="nuscenes",
                            scene=None,
                            split=None,
                            weather=None,
                        ),
                    }
                )
                token = sample["next"]
                frame_idx += 1

        return frames

    def load_sensors(self, info: dict) -> dict[str, SensorFrame]:
        """Load sensor data for a frame and return a sensor_id -> SensorFrame mapping."""
        sample = self.nusc.get("sample", info["sample_token"])
        sensors: dict[str, SensorFrame] = {}

        for channel, sample_data_token in sample["data"].items():
            sample_data = self.nusc.get("sample_data", sample_data_token)
            calibrated_sensor = self.nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
            ego = self.nusc.get("ego_pose", sample_data["ego_pose_token"])

            modality = sample_data["sensor_modality"]  # "camera" / "lidar" / "radar"
            sensor_id = channel

            # Handling Camera Frame
            if modality == "camera":
                img_path = self.nusc.get_sample_data_path(sample_data_token)
                img = cv2.imread(img_path)

                intrinsics = np.array(calibrated_sensor["camera_intrinsic"], dtype=np.float32)
                # sensor_pose_in_ego is the 4x4 transform that maps coordinates from the sensor frame into the ego (vehicle) frame.
                sensor_pose_in_ego = se3_from_quaternion(calibrated_sensor["rotation"], calibrated_sensor["translation"])

                meta = ImageMeta(spectral="rgb", frame=f"sensor:{sensor_id}", intrinsics=intrinsics, sensor_pose_in_ego=sensor_pose_in_ego)
                data = ImageSensorFrame(sensor_id=sensor_id, meta=meta, image=img, mask=None)
                sensors[sensor_id] = SensorFrame(kind="image", data=data)

            # Handling Lidar Frame
            elif modality == "lidar":
                point_cloud_path = self.nusc.get_sample_data_path(sample_data_token)
                point_cloud = NuScenesLidarPointCloud.from_file(point_cloud_path)
                xyz = point_cloud.points[:3, :].T

                sensor_pose_in_ego = se3_from_quaternion(calibrated_sensor["rotation"], calibrated_sensor["translation"])
                ego_pose_in_world = se3_from_quaternion(ego["rotation"], ego["translation"])
                meta = LidarMeta(frame=f"sensor:{sensor_id}", sensor_pose_in_ego=sensor_pose_in_ego, ego_pose_in_world=ego_pose_in_world)
                data = LidarSensorFrame(sensor_id=sensor_id, meta=meta, point_cloud=LidarPointCloud(xyz=xyz, features={}, frame="sensor"))
                sensors[sensor_id] = SensorFrame(kind="lidar", data=data)

            # Handling Radar Frame
            elif modality == "radar":
                point_cloud_path = self.nusc.get_sample_data_path(sample_data_token)
                point_cloud = RadarPointCloud.from_file(point_cloud_path)
                xyz = point_cloud.points[:3, :].T

                rcs = point_cloud.points[5, :]
                vx = point_cloud.points[6, :]
                vy = point_cloud.points[7, :]
                rng = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2) + 1e-6
                doppler = (xyz[:, 0] * vx + xyz[:, 1] * vy) / rng

                features = {
                    "rcs": rcs.astype(np.float32),
                    "vx": vx.astype(np.float32),
                    "vy": vy.astype(np.float32),
                    "doppler": doppler.astype(np.float32),
                }

                sensor_pose_in_ego = se3_from_quaternion(calibrated_sensor["rotation"], calibrated_sensor["translation"])
                ego_pose_in_world = se3_from_quaternion(ego["rotation"], ego["translation"])

                meta = RadarMeta(frame=f"sensor:{sensor_id}", sensor_pose_in_ego=sensor_pose_in_ego, ego_pose_in_world=ego_pose_in_world)

                grids = None
                if self.synthesize_radar_grids:
                    grids = self.build_radar_grids(xyz, features)

                data = RadarSensorFrame(
                    sensor_id=sensor_id,
                    meta=meta,
                    grids=grids,
                    point_cloud=PointCloud(xyz=xyz, features=features, frame="sensor"),
                )
                sensors[sensor_id] = SensorFrame(kind="radar", data=data)

        return sensors

    def load_overlays(self, info: dict) -> OverlaySet | None:
        """Load overlays/labels for a frame; return None if unlabeled."""
        sample = self.nusc.get("sample", info["sample_token"])
        boxes = []

        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)

            # ann["translation"] is center, ann["size"] is (w,l,h), ann["rotation"] is quaternion
            center_xyz = np.array(ann["translation"], dtype=np.float32)

            # Converting (w,l,h) -> (l,w,h)
            size_wlh = ann["size"]
            size_lwh = np.array([size_wlh[1], size_wlh[0], size_wlh[2]], dtype=np.float32)

            yaw = yaw_from_quaternion(ann["rotation"])

            class_id = self.class_map.get(ann["category_name"])
            if class_id is None:
                continue

            meta = OverlayMeta(coord_frame="world", source="gt", timestamp=info["timestamp"], sensor_id=None)
            boxes.append(
                Box3D(
                    meta=meta,
                    center_xyz=center_xyz,
                    size_lwh=size_lwh,
                    yaw=yaw,
                    class_id=class_id,
                    confidence=None,
                    track_id=self.instance_id_map[ann["instance_token"]],
                    velocity_xyz=None,
                )
            )

        return OverlaySet(
            boxes3D={"gt": boxes},
            boxes2D={},
            oriented_boxes2d={},
            radar_dets={},
            radar_polar_dets={},
            tracks={},
        )


# Testing the Adapter to see if everything looks good
# Can be ran from root using
#   python -m src.data.adapters.nuscenes
if __name__ == "__main__":
    nuscenes_adapter = NuScenesAdapter()

    # Checking the sensor ids
    print("--- Sensor Ids ---")
    sensor_ids = nuscenes_adapter.get_sensor_ids()
    for sensor_id in sensor_ids:
        print(sensor_id)
    print("")

    # Checking the sequence ids
    print("--- Sequence Ids ---")
    sequence_ids = nuscenes_adapter.get_sequence_ids()
    for sequence_id in sequence_ids:
        print(sequence_id)
    print("")

    # Looping through each frame
    print("--- Frames ---")
    first_frame = True
    for frame in nuscenes_adapter:  # using __getitem__ defined in BaseAdapter
        # ensuring that we only loop through the first video
        # we are doing this since frame id restarts for each video sequence
        if frame.frame_id == 0:
            if not first_frame:
                break
            first_frame = False

        print(frame.frame_id)
    print("")
