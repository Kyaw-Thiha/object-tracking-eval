from dataclasses import dataclass
from typing import List
import numpy as np

from data.schema.radar import RadarSensorFrame

from .base import BaseView
from ..schema.render_spec import RenderSpec
from ..schema.layers import PointLayer, TrackLayer
from ..schema.base_layer import Layer, LayerMeta
from ...data.schema.frame import Frame
from ...data.schema.overlay import Track


@dataclass
class RadarPointViewConfig:
    sensor_id: str
    source_key: str
    value_key: str | None = None
    units: str | None = None
    show_tracks: bool = True


class RadarPointView(BaseView[RadarPointViewConfig]):
    name = "RadarPointView"

    def build(self, frame: Frame, cfg: RadarPointViewConfig) -> RenderSpec:
        layers: List[Layer] = [self.build_point_layer(frame, cfg)]

        track_layer = self.build_tracks_layer(frame, cfg)
        if track_layer is not None:
            layers.append(track_layer)

        meta = self.build_meta(frame, [cfg.sensor_id], [cfg.source_key])
        radar = frame.sensors[cfg.sensor_id].data
        return RenderSpec(title=f"{cfg.sensor_id}:points", coord_frame=radar.meta.frame, layers=layers, meta=meta)

    def build_point_layer(self, frame: Frame, cfg: RadarPointViewConfig) -> PointLayer:
        radar = frame.sensors[cfg.sensor_id].data
        assert radar is RadarSensorFrame

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
