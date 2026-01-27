"""Radar grid view builder."""

from dataclasses import dataclass
from typing import Literal

from ...data.schema.radar import RadarSensorFrame

from .base import BaseView
from ..schema.render_spec import RenderSpec
from ..schema.layers import LineLayer, PointLayer, RasterLayer
from ..schema.base_layer import Layer, LayerMeta
from ..geometry import box3d_bev_footprint, xyz_to_radar_ra
from ..transforms import invert_se3, transform_boxes3d
from ...data.schema.frame import Frame


@dataclass
class RadarGridViewConfig:
    """Configuration for RadarGridView rendering."""

    sensor_id: str
    source_key: str
    grid_name: str
    display: Literal["pixel", "polar"] = "pixel"
    show_gt_centers: bool = False
    show_gt_footprints: bool = False


class RadarGridView(BaseView[RadarGridViewConfig]):
    """Builds a radar grid RenderSpec from RAD/RA/RD tensors."""

    name = "RadarGridView"

    def build(self, frame: Frame, cfg: RadarGridViewConfig) -> RenderSpec:
        """Assemble the radar grid view layers for a frame."""
        layers: list[Layer] = [self.build_grid_layer(frame, cfg)]
        layers.extend(self.build_gt_layers(frame, cfg))
        meta = self.build_meta(frame, [cfg.sensor_id], [cfg.source_key])

        radar = frame.sensors[cfg.sensor_id].data
        return RenderSpec(title=f"{cfg.sensor_id}:{cfg.grid_name}", coord_frame=radar.meta.frame, layers=layers, meta=meta)

    def build_grid_layer(self, frame: Frame, cfg: RadarGridViewConfig) -> RasterLayer:
        """Create the radar grid raster layer for a single grid product."""
        radar = frame.sensors[cfg.sensor_id].data
        assert isinstance(radar, RadarSensorFrame)
        if radar.grids is None or cfg.grid_name not in radar.grids:
            raise ValueError(f"Grid {cfg.grid_name} not found for {cfg.sensor_id}")
        grid = radar.grids[cfg.grid_name]

        return RasterLayer(
            name=f"{cfg.sensor_id}.{cfg.grid_name}",
            meta=LayerMeta(
                source=cfg.source_key,
                sensor_id=cfg.sensor_id,
                kind="grid",
                coord_frame=radar.meta.frame,
                timestamp=frame.timestamp,
            ),
            data=grid.tensor,
            axes=grid.axes,
            bins=grid.bins,
            extent=None,
            grid_name=cfg.grid_name,
            display=cfg.display,
        )

    def build_gt_layers(self, frame: Frame, cfg: RadarGridViewConfig) -> list[Layer]:
        """Create GT box overlays in RA grid space."""
        if cfg.grid_name != "RA":
            return []
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
        centers, sizes, yaws = transform_boxes3d(centers, sizes, yaws, T_sensor_world)

        layers: list[Layer] = []
        if cfg.show_gt_centers:
            ra = xyz_to_radar_ra(centers)
            layers.append(
                PointLayer(
                    name=f"{cfg.sensor_id}.gt_centers.ra",
                    meta=LayerMeta(
                        source="gt",
                        sensor_id=cfg.sensor_id,
                        kind="gt_center",
                        coord_frame="grid:ra",
                        timestamp=frame.timestamp,
                    ),
                    xyz=np.stack([ra[:, 1], ra[:, 0]], axis=1),
                    value=None,
                    color=None,
                    value_key=None,
                    units=None,
                )
            )

        if cfg.show_gt_footprints:
            segments = []
            for center, size, yaw in zip(centers, sizes, yaws):
                corners = box3d_bev_footprint(center, size, yaw)
                ra = xyz_to_radar_ra(np.column_stack([corners, np.zeros((corners.shape[0], 1))]))
                seg = np.stack([ra[:-1], ra[1:]], axis=1)
                seg = seg[:, :, [1, 0]]
                segments.append(seg)
            if segments:
                segments_np = np.concatenate(segments, axis=0)
                layers.append(
                    LineLayer(
                        name=f"{cfg.sensor_id}.gt_boxes.ra",
                        meta=LayerMeta(
                            source="gt",
                            sensor_id=cfg.sensor_id,
                            kind="gt_box",
                            coord_frame="grid:ra",
                            timestamp=frame.timestamp,
                        ),
                        segments=segments_np,
                    )
                )

        return layers
