"""Radar point view builder."""

from dataclasses import dataclass
import numpy as np

from ...data.schema.radar import RadarSensorFrame

from .base import BaseView
from ..schema.render_spec import RenderSpec
from ..schema.layers import LineLayer, PointLayer, TrackLayer
from ..schema.base_layer import Layer, LayerMeta
from ..geometry import box3d_bev_footprint
from ..palette import CLASS_COLORS, DEFAULT_COLOR
from ..schema.base_layer import LayerStyle
from ..transforms import invert_se3, transform_boxes3d
from ...data.schema.frame import Frame
from ...data.schema.overlay import Track


@dataclass
class RadarPointViewConfig:
    """Configuration for RadarPointView rendering."""

    sensor_id: str
    source_key: str
    value_key: str | None = None
    units: str | None = None
    show_tracks: bool = True
    show_gt_centers: bool = False
    show_gt_footprints: bool = False


class RadarPointView(BaseView[RadarPointViewConfig]):
    """Builds a radar point RenderSpec with optional track overlays."""

    name = "RadarPointView"

    def build(self, frame: Frame, cfg: RadarPointViewConfig) -> RenderSpec:
        """Assemble the radar point view layers for a frame."""
        layers: list[Layer] = [self.build_point_layer(frame, cfg)]
        layers.extend(self.build_gt_layers(frame, cfg))

        track_layer = self.build_tracks_layer(frame, cfg)
        if track_layer is not None:
            layers.append(track_layer)

        meta = self.build_meta(frame, [cfg.sensor_id], [cfg.source_key])
        radar = frame.sensors[cfg.sensor_id].data
        return RenderSpec(title=f"{cfg.sensor_id}:points", coord_frame=radar.meta.frame, layers=layers, meta=meta)

    def build_point_layer(self, frame: Frame, cfg: RadarPointViewConfig) -> PointLayer:
        """Create the point cloud layer for radar detections."""
        radar = frame.sensors[cfg.sensor_id].data
        assert isinstance(radar, RadarSensorFrame)

        pc = radar.point_cloud
        if pc is None:
            raise ValueError(f"Point cloud not found for {cfg.sensor_id}")

        value = pc.features.get(cfg.value_key) if (cfg.value_key and pc.features) else None
        return PointLayer(
            name=f"{cfg.sensor_id}.points",
            meta=LayerMeta(
                source=cfg.source_key,
                sensor_id=cfg.sensor_id,
                kind="pc",
                coord_frame=radar.meta.frame,
                timestamp=frame.timestamp,
            ),
            xyz=pc.xyz,
            value=value,
            color=None,
            value_key=cfg.value_key,
            units=cfg.units,
        )

    def build_tracks_layer(self, frame: Frame, cfg: RadarPointViewConfig) -> TrackLayer | None:
        """Create the track overlay layer if available."""
        if not (cfg.show_tracks and frame.overlays and cfg.source_key in frame.overlays.tracks):
            return None

        tracks = [t for t in frame.overlays.tracks[cfg.source_key] if isinstance(t, Track) and t.meta.sensor_id == cfg.sensor_id]
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
                coord_frame=frame.sensors[cfg.sensor_id].data.meta.frame,
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

    def build_gt_layers(self, frame: Frame, cfg: RadarPointViewConfig) -> list[Layer]:
        """Create GT box overlays in radar frame."""
        if not (frame.overlays and cfg.source_key in frame.overlays.boxes3D):
            return []
        if not (cfg.show_gt_centers or cfg.show_gt_footprints):
            return []

        radar = frame.sensors[cfg.sensor_id].data
        if radar.meta.ego_pose_in_world is None:
            return []

        T_ego_world = invert_se3(radar.meta.ego_pose_in_world)
        T_sensor_ego = invert_se3(radar.meta.sensor_pose_in_ego)
        T_sensor_world = T_sensor_ego @ T_ego_world

        boxes = [b for b in frame.overlays.boxes3D[cfg.source_key] if b.meta.coord_frame == "world"]
        if not boxes:
            return []

        centers = np.stack([b.center_xyz for b in boxes], axis=0)
        sizes = np.stack([b.size_lwh for b in boxes], axis=0)
        yaws = np.array([b.yaw for b in boxes], dtype=float)
        class_ids = np.array([b.class_id for b in boxes], dtype=int)
        centers, sizes, yaws = transform_boxes3d(centers, sizes, yaws, T_sensor_world)

        layers: list[Layer] = []
        if cfg.show_gt_centers:
            colors = np.array([CLASS_COLORS.get(int(cid), DEFAULT_COLOR) for cid in class_ids], dtype=float)
            layers.append(
                PointLayer(
                    name=f"{cfg.sensor_id}.gt_centers",
                    meta=LayerMeta(
                        source="gt",
                        sensor_id=cfg.sensor_id,
                        kind="gt_center",
                        coord_frame=radar.meta.frame,
                        timestamp=frame.timestamp,
                    ),
                    xyz=centers,
                    value=None,
                    color=colors,
                    value_key=None,
                    units=None,
                )
            )

        if cfg.show_gt_footprints:
            segments_by_class: dict[int, list[np.ndarray]] = {}
            for center, size, yaw, class_id in zip(centers, sizes, yaws, class_ids):
                corners = box3d_bev_footprint(center, size, yaw)
                seg = np.stack([corners[:-1], corners[1:]], axis=1)
                segments_by_class.setdefault(int(class_id), []).append(seg)
            for class_id, segments in segments_by_class.items():
                segments_np = np.concatenate(segments, axis=0)
                color = CLASS_COLORS.get(int(class_id), DEFAULT_COLOR)
                layers.append(
                    LineLayer(
                        name=f"{cfg.sensor_id}.gt_boxes.{class_id}",
                        meta=LayerMeta(
                            source="gt",
                            sensor_id=cfg.sensor_id,
                            kind="gt_box",
                            coord_frame=radar.meta.frame,
                            timestamp=frame.timestamp,
                        ),
                        segments=segments_np,
                        style=LayerStyle(color=color, line_width=1.5),
                    )
                )

        return layers
