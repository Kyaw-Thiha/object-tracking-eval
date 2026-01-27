"""Radar grid view builder."""

from dataclasses import dataclass
import numpy as np
from typing import Literal

from ...data.schema.radar import RadarSensorFrame

from .base import BaseView
from ..schema.render_spec import RenderSpec
from ..schema.layers import LineLayer, PointLayer, RasterLayer
from ..schema.base_layer import Layer, LayerMeta
from ..geometry import box3d_bev_footprint, xyz_to_radar_ra
from ..palette import CLASS_COLORS, DEFAULT_COLOR
from ..schema.base_layer import LayerStyle
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
        if not (frame.overlays and cfg.source_key in frame.overlays.boxes3D):
            return []
        if not (cfg.show_gt_centers or cfg.show_gt_footprints):
            return []

        radar = frame.sensors[cfg.sensor_id].data
        if radar.grids is None or cfg.grid_name not in radar.grids:
            return []
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

        grid = radar.grids[cfg.grid_name]
        range_bins = grid.bins.get("range") if grid.bins else None
        if range_bins is None:
            return []
        r_min, r_max = float(range_bins.min()), float(range_bins.max())

        layers: list[Layer] = []
        if cfg.grid_name == "RA":
            azimuth_bins = grid.bins.get("azimuth") if grid.bins else None
            if azimuth_bins is None:
                return layers
            a_min, a_max = float(azimuth_bins.min()), float(azimuth_bins.max())

            if cfg.show_gt_centers:
                ra = xyz_to_radar_ra(centers)
                mask = (ra[:, 0] >= r_min) & (ra[:, 0] <= r_max) & (ra[:, 1] >= a_min) & (ra[:, 1] <= a_max)
                ra = ra[mask]
                class_ids_ra = class_ids[mask]
                if ra.size > 0:
                    colors = np.array([CLASS_COLORS.get(int(cid), DEFAULT_COLOR) for cid in class_ids_ra], dtype=float)
                    if cfg.display == "polar":
                        coords = ra
                        coord_frame = "grid:ra:polar"
                    else:
                        coords = np.stack([ra[:, 1], ra[:, 0]], axis=1)
                        coord_frame = "grid:ra"
                    layers.append(
                        PointLayer(
                            name=f"{cfg.sensor_id}.gt_centers.ra",
                            meta=LayerMeta(
                                source="gt",
                                sensor_id=cfg.sensor_id,
                                kind="gt_center",
                                coord_frame=coord_frame,
                                timestamp=frame.timestamp,
                            ),
                            xyz=coords,
                            value=None,
                            color=colors,
                            value_key=None,
                            units=None,
                        )
                    )

            if cfg.show_gt_footprints:
                segments_by_class: dict[int, list[np.ndarray]] = {cid: [] for cid in sorted(CLASS_COLORS.keys())}
                for center, size, yaw, class_id in zip(centers, sizes, yaws, class_ids):
                    corners = box3d_bev_footprint(center, size, yaw)
                    ra = xyz_to_radar_ra(np.column_stack([corners, np.zeros((corners.shape[0], 1))]))
                    in_bounds = (ra[:, 0] >= r_min) & (ra[:, 0] <= r_max) & (ra[:, 1] >= a_min) & (ra[:, 1] <= a_max)
                    if not np.all(in_bounds):
                        continue
                    if cfg.display == "polar":
                        seg = np.stack([ra[:-1], ra[1:]], axis=1)
                    else:
                        seg = np.stack([ra[:-1], ra[1:]], axis=1)
                        seg = seg[:, :, [1, 0]]
                    segments_by_class.setdefault(int(class_id), []).append(seg)
                if segments_by_class:
                    coord_frame = "grid:ra:polar" if cfg.display == "polar" else "grid:ra"
                    for class_id in segments_by_class:
                        segments = segments_by_class[class_id]
                        if segments:
                            segments_np = np.concatenate(segments, axis=0)
                        else:
                            segments_np = np.empty((0, 2, 2), dtype=float)
                        color = CLASS_COLORS.get(int(class_id), DEFAULT_COLOR)
                        layers.append(
                            LineLayer(
                                name=f"{cfg.sensor_id}.gt_boxes.ra.{class_id}",
                                meta=LayerMeta(
                                    source="gt",
                                    sensor_id=cfg.sensor_id,
                                    kind="gt_box",
                                    coord_frame=coord_frame,
                                    timestamp=frame.timestamp,
                                ),
                                segments=segments_np,
                                style=LayerStyle(color=color, line_width=1.5),
                            )
                        )

            return layers

        if cfg.grid_name == "RD":
            doppler_bins = grid.bins.get("doppler") if grid.bins else None
            if doppler_bins is None:
                return layers
            d_min, d_max = float(doppler_bins.min()), float(doppler_bins.max())

            velocities = []
            centers_valid = []
            class_ids_valid = []
            for center, class_id, box in zip(centers, class_ids, boxes):
                if box.velocity_xyz is None:
                    continue
                vel_world = box.velocity_xyz
                if radar.meta.ego_velocity_in_world is not None:
                    vel_world = vel_world - radar.meta.ego_velocity_in_world
                centers_valid.append(center)
                class_ids_valid.append(class_id)
                velocities.append(vel_world)

            if not centers_valid:
                coords = np.empty((0, 2), dtype=float)
                colors = np.empty((0, 3), dtype=float)
                layers.append(
                    PointLayer(
                        name=f"{cfg.sensor_id}.gt_centers.rd",
                        meta=LayerMeta(
                            source="gt",
                            sensor_id=cfg.sensor_id,
                            kind="gt_center",
                            coord_frame="grid:rd",
                            timestamp=frame.timestamp,
                        ),
                        xyz=coords,
                        value=None,
                        color=colors,
                        value_key=None,
                        units=None,
                    )
                )
                return layers

            centers_valid = np.stack(centers_valid, axis=0)
            class_ids_valid = np.array(class_ids_valid, dtype=int)
            velocities_world = np.stack(velocities, axis=0)

            R = T_sensor_world[:3, :3]
            velocities_sensor = (R @ velocities_world.T).T

            xy = centers_valid[:, :2]
            ranges = np.linalg.norm(xy, axis=1) + 1e-6
            doppler = (velocities_sensor[:, 0] * xy[:, 0] + velocities_sensor[:, 1] * xy[:, 1]) / ranges

            mask = (ranges >= r_min) & (ranges <= r_max) & (doppler >= d_min) & (doppler <= d_max)
            if np.any(mask):
                ranges = ranges[mask]
                doppler = doppler[mask]
                class_ids_valid = class_ids_valid[mask]
                colors = np.array([CLASS_COLORS.get(int(cid), DEFAULT_COLOR) for cid in class_ids_valid], dtype=float)
                coords = np.stack([doppler, ranges], axis=1)
            else:
                coords = np.empty((0, 2), dtype=float)
                colors = np.empty((0, 3), dtype=float)

            layers.append(
                PointLayer(
                    name=f"{cfg.sensor_id}.gt_centers.rd",
                    meta=LayerMeta(
                        source="gt",
                        sensor_id=cfg.sensor_id,
                        kind="gt_center",
                        coord_frame="grid:rd",
                        timestamp=frame.timestamp,
                    ),
                    xyz=coords,
                    value=None,
                    color=colors,
                    value_key=None,
                    units=None,
                )
            )

            return layers

        return layers
