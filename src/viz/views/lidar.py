"""Lidar view builder."""

from dataclasses import dataclass
import numpy as np

from ...data.schema.lidar import LidarSensorFrame

from .base import BaseView
from ..schema.render_spec import RenderSpec
from ..schema.layers import PointLayer, Box3DLayer, TrackLayer
from ..schema.base_layer import LayerMeta
from ...data.schema.frame import Frame
from ...data.schema.overlay import Track


@dataclass
class LidarViewConfig:
    """Configuration for LidarView rendering."""

    sensor_id: str
    source_key: str
    show_tracks: bool = True


class LidarView(BaseView[LidarViewConfig]):
    """Builds a lidar RenderSpec with points, boxes, and tracks."""

    name = "LidarView"

    def build(self, frame: Frame, cfg: LidarViewConfig) -> RenderSpec:
        """Assemble the lidar view layers for a frame."""
        layers = []

        point_layer = self.build_points_layer(frame, cfg)
        if point_layer is not None:
            layers.append(point_layer)

        box_layer = self.build_boxes3d_layer(frame, cfg)
        if box_layer is not None:
            layers.append(box_layer)

        track_layer = self.build_tracks_layer(frame, cfg)
        if track_layer is not None:
            layers.append(track_layer)

        meta = self.build_meta(frame, [cfg.sensor_id], [cfg.source_key])
        lidar = frame.sensors[cfg.sensor_id].data
        return RenderSpec(title=cfg.sensor_id, coord_frame=lidar.meta.frame, layers=layers, meta=meta)

    def build_points_layer(self, frame: Frame, cfg: LidarViewConfig) -> PointLayer | None:
        """Create the point cloud layer if available."""
        lidar = frame.sensors[cfg.sensor_id].data
        assert lidar is LidarSensorFrame
        pc = lidar.point_cloud
        if pc is None:
            return None

        return PointLayer(
            name=f"{cfg.sensor_id}.points",
            meta=LayerMeta(
                source=cfg.source_key,
                sensor_id=cfg.sensor_id,
                kind="pc",
                coord_frame=lidar.meta.frame,
                timestamp=frame.timestamp,
            ),
            xyz=pc.xyz,
            value=None,
            color=None,
            value_key=None,
            units=None,
        )

    def build_boxes3d_layer(self, frame: Frame, cfg: LidarViewConfig) -> Box3DLayer | None:
        """Create the 3D box overlay layer if available."""
        lidar = frame.sensors[cfg.sensor_id].data

        if not (frame.overlays and cfg.source_key in frame.overlays.boxes3D):
            return None

        boxes = frame.overlays.boxes3D[cfg.source_key]
        if not boxes:
            return None

        centers = np.stack([b.center_xyz for b in boxes], axis=0)
        sizes = np.stack([b.size_lwh for b in boxes], axis=0)
        yaws = np.array([b.yaw for b in boxes], dtype=float)

        return Box3DLayer(
            name=f"{cfg.sensor_id}.boxes3d",
            meta=LayerMeta(
                source=cfg.source_key,
                sensor_id=cfg.sensor_id,
                kind="bbox3d",
                coord_frame=lidar.meta.frame,
                timestamp=frame.timestamp,
            ),
            centers=centers,
            sizes_lwh=sizes,
            yaws=yaws,
            labels=None,
            class_ids=None,
        )

    def build_tracks_layer(self, frame: Frame, cfg: LidarViewConfig) -> TrackLayer | None:
        """Create the track overlay layer if available."""
        lidar = frame.sensors[cfg.sensor_id].data

        if not (cfg.show_tracks and frame.overlays and cfg.source_key in frame.overlays.tracks):
            return None

        tracks = [t for t in frame.overlays.tracks[cfg.source_key] if isinstance(t, Track)]
        if not tracks:
            return None

        positions = np.stack([t.states[-1].position_xyz for t in tracks], axis=0)
        track_ids = np.array([t.track_id for t in tracks], dtype=int)

        return TrackLayer(
            name=f"{cfg.sensor_id}.tracks",
            meta=LayerMeta(
                source=cfg.source_key,
                sensor_id=cfg.sensor_id,
                kind="track",
                coord_frame=lidar.meta.frame,
                timestamp=frame.timestamp,
            ),
            track_ids=track_ids,
            positions_xyz=positions,
            velocities_xyz=None,
            yaws=None,
            covariances=None,
            labels=None,
            history=None,
            source_key=cfg.source_key,
            history_len=None,
            velocity_units=None,
        )
