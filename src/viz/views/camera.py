"""Camera view builder."""

from dataclasses import dataclass
import numpy as np

from .base import BaseView
from ..schema.render_spec import RenderSpec
from ..schema.layers import RasterLayer, Box2DLayer, TrackLayer, TextLayer
from ..schema.base_layer import Layer, LayerMeta, LayerStyle
from ..transforms import invert_se3, project_points_to_image, transform_points
from ..geometry import box3d_to_box2d, ego_pose_in_world_from_frame
from ..palette import CLASS_COLORS
from ...data.schema.frame import Frame
from ...data.schema.image import ImageSensorFrame
from ...data.schema.overlay import Box2D, Track


@dataclass
class CameraViewConfig:
    """Configuration for CameraView rendering."""

    sensor_id: str
    source_key: str
    show_boxes: bool = True
    show_labels: bool = True
    show_tracks: bool = True


class CameraView(BaseView[CameraViewConfig]):
    """Builds a camera RenderSpec with image, 2D boxes, and tracks."""

    name = "CameraView"

    def build(self, frame: Frame, cfg: CameraViewConfig) -> RenderSpec:
        """Assemble the camera view layers for a frame."""
        layers: list[Layer] = [self.build_image_layer(frame, cfg)]

        box_layer = self.build_boxes2d_layer(frame, cfg)
        if box_layer is not None:
            layers.append(box_layer)

        layers.extend(self.build_tracks_layers(frame, cfg))

        meta = self.build_meta(frame, [cfg.sensor_id], [cfg.source_key])
        return RenderSpec(title=cfg.sensor_id, coord_frame=f"sensor:{cfg.sensor_id}", layers=layers, meta=meta)

    def build_image_layer(self, frame: Frame, cfg: CameraViewConfig) -> RasterLayer:
        """Create the image raster layer for the camera."""
        sensor = frame.sensors[cfg.sensor_id].data
        assert isinstance(sensor, ImageSensorFrame)

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
        """Create the 2D box overlay layer if available."""
        if not cfg.show_boxes or not frame.overlays:
            return None

        boxes = []
        if cfg.source_key in frame.overlays.boxes2D:
            for box in frame.overlays.boxes2D[cfg.source_key]:
                if isinstance(box, Box2D) and box.meta.sensor_id == cfg.sensor_id:
                    boxes.append(box)
        if len(boxes) == 0:
            if not (frame.overlays and cfg.source_key in frame.overlays.boxes3D):
                return None

            sensor = frame.sensors[cfg.sensor_id].data
            assert isinstance(sensor, ImageSensorFrame)

            ego_pose_in_world = ego_pose_in_world_from_frame(frame)
            if ego_pose_in_world is None:
                return None

            T_ego_world = invert_se3(ego_pose_in_world)
            T_sensor_ego = invert_se3(sensor.meta.sensor_pose_in_ego)
            T_cam_world = T_sensor_ego @ T_ego_world

            image_shape = (int(sensor.image.shape[0]), int(sensor.image.shape[1]))
            xyxy_list = []
            labels = []
            class_ids = []
            for b in frame.overlays.boxes3D[cfg.source_key]:
                if b.meta.coord_frame != "world":
                    continue
                rect = box3d_to_box2d(
                    center_xyz=b.center_xyz,
                    size_lwh=b.size_lwh,
                    yaw=b.yaw,
                    T_cam_world=T_cam_world,
                    intrinsics=sensor.meta.intrinsics,
                    image_shape=image_shape,
                )
                if rect is None:
                    continue
                xyxy_list.append(rect)
                class_ids.append(b.class_id)
                labels.append(f"id={b.track_id}" if b.track_id is not None else "")

            if not xyxy_list:
                return None

            xyxy = np.array(xyxy_list, dtype=np.float32)
            return Box2DLayer(
                name=f"{cfg.sensor_id}.boxes2d",
                meta=LayerMeta(
                    source=cfg.source_key,
                    sensor_id=cfg.sensor_id,
                    kind="bbox2d",
                    coord_frame="pixel",
                    timestamp=frame.timestamp,
                ),
                xyxy=xyxy,
                labels=labels if cfg.show_labels else None,
                class_ids=np.array(class_ids, dtype=int) if class_ids else None,
                style=LayerStyle(palette=CLASS_COLORS, line_width=1.5),
            )

        xyxy = np.stack([b.xyxy for b in boxes], axis=0)
        labels = [f"id={b.track_id}" if b.track_id is not None else "" for b in boxes]
        class_ids = np.array([b.class_id for b in boxes], dtype=int) if boxes else None

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
            class_ids=class_ids,
            style=LayerStyle(palette=CLASS_COLORS, line_width=1.5),
        )

    def build_tracks_layers(self, frame: Frame, cfg: CameraViewConfig) -> list[Layer]:
        """Create track and label layers if available."""
        if not (cfg.show_tracks and frame.overlays and cfg.source_key in frame.overlays.tracks):
            return []

        tracks = [
            t
            for t in frame.overlays.tracks[cfg.source_key]
            if isinstance(t, Track) and t.meta.sensor_id == cfg.sensor_id and t.states
        ]
        if not tracks:
            return []

        positions = np.stack([t.states[-1].position_xyz for t in tracks], axis=0)
        track_ids = np.array([t.track_id for t in tracks], dtype=int)

        velocities_list = [t.states[-1].velocity_xyz for t in tracks if t.states[-1].velocity_xyz is not None]
        velocities = np.stack(velocities_list, axis=0) if len(velocities_list) == len(tracks) else None

        if all(t.states[-1].yaw is not None for t in tracks):
            yaws = np.array([t.states[-1].yaw for t in tracks], dtype=float)
        else:
            yaws = None

        labels = [f"id={t.track_id}" for t in tracks] if cfg.show_labels else None

        sensor = frame.sensors[cfg.sensor_id].data
        assert isinstance(sensor, ImageSensorFrame)
        coord_frame = sensor.meta.frame
        positions_sensor = positions

        if all(t.meta.coord_frame == "ego" for t in tracks):
            sensor_pose_in_ego = sensor.meta.sensor_pose_in_ego
            if sensor_pose_in_ego is not None:
                ego_to_sensor = invert_se3(sensor_pose_in_ego)
                positions_sensor = transform_points(positions, ego_to_sensor)
                coord_frame = sensor.meta.frame
        elif all(t.meta.coord_frame == sensor.meta.frame for t in tracks):
            coord_frame = sensor.meta.frame
        else:
            coord_frame = tracks[0].meta.coord_frame

        text_layer = None
        if cfg.show_labels and coord_frame == sensor.meta.frame:
            image_shape = (int(sensor.image.shape[0]), int(sensor.image.shape[1]))
            uv, valid = project_points_to_image(positions_sensor, sensor.meta.intrinsics, image_shape)
            if not np.any(valid):
                return []

            positions_sensor = positions_sensor[valid]
            track_ids = track_ids[valid]
            if velocities is not None:
                velocities = velocities[valid]
            if yaws is not None:
                yaws = yaws[valid]
            if labels is not None:
                labels = [label for label, keep in zip(labels, valid) if keep]

            text_layer = TextLayer(
                name=f"{cfg.sensor_id}.track_labels",
                meta=LayerMeta(
                    source=cfg.source_key,
                    sensor_id=cfg.sensor_id,
                    kind="track_label",
                    coord_frame="pixel",
                    timestamp=frame.timestamp,
                ),
                xy=uv[valid],
                texts=labels or [],
            )

        layers: list[Layer] = []
        layers.append(
            TrackLayer(
                name=f"{cfg.sensor_id}.tracks",
                meta=LayerMeta(
                    source=cfg.source_key,
                    sensor_id=cfg.sensor_id,
                    kind="track",
                    coord_frame=coord_frame,
                    timestamp=frame.timestamp,
                ),
                track_ids=track_ids,
                positions_xyz=positions_sensor,
                velocities_xyz=velocities,
                yaws=yaws,
                covariances=None,
                labels=labels,
                history=None,
                source_key=cfg.source_key,
                history_len=None,
                velocity_units="m/s",
            )
        )

        if text_layer is not None:
            layers.append(text_layer)

        return layers
