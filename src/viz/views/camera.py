from dataclasses import dataclass
from typing import Union
import numpy as np

from data.schema.image import ImageSensorFrame

from .base import BaseView
from ..schema.render_spec import RenderSpec
from ..schema.layers import RasterLayer, Box2DLayer, TrackLayer
from ..schema.base_layer import Layer, LayerMeta
from ...data.schema.frame import Frame
from ...data.schema.overlay import Box2D, Track


@dataclass
class CameraViewConfig:
    sensor_id: str
    source_key: str
    show_boxes: bool = True
    show_labels: bool = True
    show_tracks: bool = True


class CameraView(BaseView[CameraViewConfig]):
    name = "CameraView"

    def build(self, frame: Frame, cfg: CameraViewConfig) -> RenderSpec:
        layers: list[Layer] = [self.build_image_layer(frame, cfg)]

        box_layer = self.build_boxes2d_layer(frame, cfg)
        if box_layer is not None:
            layers.append(box_layer)

        track_layer = self.build_tracks_layer(frame, cfg)
        if track_layer is not None:
            layers.append(track_layer)

        meta = self.build_meta(frame, [cfg.sensor_id], [cfg.source_key])
        return RenderSpec(title=cfg.sensor_id, coord_frame=f"sensor:{cfg.sensor_id}", layers=layers, meta=meta)

    def build_image_layer(self, frame: Frame, cfg: CameraViewConfig) -> RasterLayer:
        sensor = frame.sensors[cfg.sensor_id].data
        assert sensor is ImageSensorFrame

        return RasterLayer(
            name=f"{cfg.sensor_id}.image",
            meta=LayerMeta(
                source=cfg.source_key,
                sensor_id=cfg.sensor_id,
                kind="image",
                coord_frame=f"sensor:{cfg.sensor_id}",
                timestamp=frame.timestamp,
            ),
            data=sensor.image,
            axes=None,
            bins=None,
            extent=None,
            grid_name=None,
            display="pixel",
        )

    def build_boxes2d_layer(self, frame: Frame, cfg: CameraViewConfig) -> Box2DLayer | None:
        if not (cfg.show_boxes and frame.overlays and cfg.source_key in frame.overlays.boxes2D):
            return None

        boxes = []
        for box in frame.overlays.boxes2D[cfg.source_key]:
            if isinstance(box, Box2D) and box.meta.sensor_id == cfg.sensor_id:
                boxes.append(box)
        if len(boxes) == 0:
            return None

        xyxy = np.stack([b.xyxy for b in boxes], axis=0)
        labels = [f"id={b.track_id}" if b.track_id is not None else "" for b in boxes]

        return Box2DLayer(
            name=f"{cfg.sensor_id}.boxes2d",
            meta=LayerMeta(
                source=cfg.source_key,
                sensor_id=cfg.sensor_id,
                kind="bbox2d",
                coord_frame=f"sensor:{cfg.sensor_id}",
                timestamp=frame.timestamp,
            ),
            xyxy=xyxy,
            labels=labels if cfg.show_labels else None,
            class_ids=None,
        )

    def build_tracks_layer(self, frame: Frame, cfg: CameraViewConfig) -> TrackLayer | None:
        if not (cfg.show_tracks and frame.overlays and cfg.source_key in frame.overlays.tracks):
            return None

        tracks = [t for t in frame.overlays.tracks[cfg.source_key] if isinstance(t, Track) and t.meta.sensor_id == cfg.sensor_id]
        if not tracks:
            return None

        # NOTE: No 2D projection available yet.
        # If you later add image-space projections, plug them here.
        positions = np.stack([t.states[-1].position_xyz for t in tracks], axis=0)
        track_ids = np.array([t.track_id for t in tracks], dtype=int)
        velocities = (
            np.stack([t.states[-1].velocity_xyz for t in tracks if t.states[-1].velocity_xyz is not None], axis=0)
            if any(t.states[-1].velocity_xyz is not None for t in tracks)
            else None
        )
        yaws = (
            np.array([t.states[-1].yaw for t in tracks if t.states[-1].yaw is not None], dtype=float)
            if any(t.states[-1].yaw is not None for t in tracks)
            else None
        )

        labels = [f"id={t.track_id}" for t in tracks] if cfg.show_labels else None

        return TrackLayer(
            name=f"{cfg.sensor_id}.tracks",
            meta=LayerMeta(
                source=cfg.source_key,
                sensor_id=cfg.sensor_id,
                kind="track",
                coord_frame=f"sensor:{cfg.sensor_id}",
                timestamp=frame.timestamp,
            ),
            track_ids=track_ids,
            positions_xyz=positions,
            velocities_xyz=velocities,
            yaws=yaws,
            covariances=None,
            labels=labels,
            history=None,
            source_key=cfg.source_key,
            history_len=None,
            velocity_units="m/s",
        )
