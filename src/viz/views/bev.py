"""BEV view builder."""

from dataclasses import dataclass
import math
import numpy as np

from ...data.schema.lidar import LidarSensorFrame
from ...data.schema.radar import RadarSensorFrame

from .base import BaseView
from ..schema.render_spec import RenderSpec
from ..schema.layers import PointLayer, LineLayer, Box3DLayer, TrackLayer
from ..schema.base_layer import LayerMeta
from ..transforms import invert_se3, transform_boxes3d
from ..geometry import ego_pose_in_world_from_frame
from ...data.schema.frame import Frame
from ...data.schema.overlay import Track


@dataclass
class BEVViewConfig:
    """Configuration for BEVView rendering."""

    source_keys: list[str]
    sensor_ids: list[str]
    max_points: int | None = None
    show_axes: bool = True
    show_tracks: bool = True


class BEVView(BaseView[BEVViewConfig]):
    """Builds a BEV RenderSpec from multiple sensor inputs and overlays."""

    name = "BEVView"

    def build(self, frame: Frame, cfg: BEVViewConfig) -> RenderSpec:
        """Assemble the BEV view layers for a frame."""
        layers = []

        if cfg.show_axes:
            layers.append(self.build_axes_layer(frame))

        layers.extend(self.build_sensor_point_layers(frame, cfg))
        layers.extend(self.build_boxes3d_layers(frame, cfg))

        track_layers = self.build_tracks_layers(frame, cfg)
        layers.extend(track_layers)

        meta = self.build_meta(frame, cfg.sensor_ids, cfg.source_keys)
        return RenderSpec(title="BEV", coord_frame="ego", layers=layers, meta=meta)

    def build_axes_layer(self, frame: Frame) -> LineLayer:
        """Create ego-axis line layers."""
        axes = np.array(
            [
                [[0, 0, 0], [5, 0, 0]],
                [[0, 0, 0], [0, 5, 0]],
            ],
            dtype=float,
        )

        return LineLayer(
            name="ego.axes",
            meta=LayerMeta(
                source="system",
                sensor_id=None,
                kind="axes",
                coord_frame="ego",
                timestamp=frame.timestamp,
            ),
            segments=axes,
        )

    def build_sensor_point_layers(self, frame: Frame, cfg: BEVViewConfig) -> list[PointLayer]:
        """Create point cloud layers for sensors that provide point clouds."""
        layers = []
        for sensor_id in cfg.sensor_ids:
            sensor = frame.sensors[sensor_id].data
            if isinstance(sensor, (LidarSensorFrame, RadarSensorFrame)):
                if sensor.point_cloud is not None:  # Since radar can have empty point cloud
                    xyz = sensor.point_cloud.xyz
                    if cfg.max_points is not None and xyz.shape[0] > cfg.max_points:
                        stride = max(1, math.ceil(xyz.shape[0] / cfg.max_points))
                        xyz = xyz[::stride]
                    layers.append(
                        PointLayer(
                            name=f"{sensor_id}.points",
                            meta=LayerMeta(
                                source="sensor",
                                sensor_id=sensor_id,
                                kind="pc",
                                coord_frame="ego",
                                timestamp=frame.timestamp,
                            ),
                            xyz=xyz,
                            value=None,
                            color=None,
                            value_key=None,
                            units=None,
                        )
                    )
        return layers

    def build_boxes3d_layers(self, frame: Frame, cfg: BEVViewConfig) -> list[Box3DLayer]:
        """Create 3D box overlay layers for each requested source."""
        layers = []
        if not frame.overlays:
            return layers

        for source_key in cfg.source_keys:
            boxes = frame.overlays.boxes3D.get(source_key, [])
            if not boxes:
                continue

            boxes_world = [b for b in boxes if b.meta.coord_frame == "world"]
            boxes_ego = [b for b in boxes if b.meta.coord_frame == "ego"]

            centers_list = []
            sizes_list = []
            yaws_list = []

            if boxes_world:
                ego_pose_in_world = ego_pose_in_world_from_frame(frame)
                if ego_pose_in_world is not None:
                    centers = np.stack([b.center_xyz for b in boxes_world], axis=0)
                    sizes = np.stack([b.size_lwh for b in boxes_world], axis=0)
                    yaws = np.array([b.yaw for b in boxes_world], dtype=float)
                    centers, sizes, yaws = transform_boxes3d(centers, sizes, yaws, invert_se3(ego_pose_in_world))
                    centers_list.append(centers)
                    sizes_list.append(sizes)
                    yaws_list.append(yaws)

            if boxes_ego:
                centers_list.append(np.stack([b.center_xyz for b in boxes_ego], axis=0))
                sizes_list.append(np.stack([b.size_lwh for b in boxes_ego], axis=0))
                yaws_list.append(np.array([b.yaw for b in boxes_ego], dtype=float))

            if not centers_list:
                continue

            centers = np.concatenate(centers_list, axis=0)
            sizes = np.concatenate(sizes_list, axis=0)
            yaws = np.concatenate(yaws_list, axis=0)

            layers.append(
                Box3DLayer(
                    name=f"boxes3d.{source_key}",
                    meta=LayerMeta(
                        source=source_key,
                        sensor_id=None,
                        kind="bbox3d",
                        coord_frame="ego",
                        timestamp=frame.timestamp,
                    ),
                    centers=centers,
                    sizes_lwh=sizes,
                    yaws=yaws,
                    labels=None,
                    class_ids=None,
                )
            )
        return layers

    def build_tracks_layers(self, frame: Frame, cfg: BEVViewConfig) -> list[TrackLayer]:
        """Create track overlay layers for each requested source."""
        layers = []
        if not (cfg.show_tracks and frame.overlays):
            return layers

        for source_key in cfg.source_keys:
            tracks = frame.overlays.tracks.get(source_key, [])
            if not tracks:
                continue

            positions = np.stack([t.states[-1].position_xyz for t in tracks], axis=0)
            track_ids = np.array([t.track_id for t in tracks], dtype=int)

            layers.append(
                TrackLayer(
                    name=f"tracks.{source_key}",
                    meta=LayerMeta(
                        source=source_key,
                        sensor_id=None,
                        kind="track",
                        coord_frame="ego",
                        timestamp=frame.timestamp,
                    ),
                    track_ids=track_ids,
                    positions_xyz=positions,
                    velocities_xyz=None,
                    yaws=None,
                    covariances=None,
                    labels=None,
                    history=None,
                    source_key=source_key,
                    history_len=None,
                    velocity_units=None,
                )
            )
        return layers
