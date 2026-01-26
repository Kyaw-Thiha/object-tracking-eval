"""Open3D backend for RenderSpec visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import os

import numpy as np
import open3d as o3d

from .base import BaseBackend
from ..schema.render_spec import RenderSpec
from ..schema.layers import PointLayer, LineLayer, Box3DLayer, TrackLayer


@dataclass
class Open3DHandle:
    vis: Any
    geometries: list[o3d.geometry.Geometry]


class Open3DBackend(BaseBackend):
    """Render 3D layers into an Open3D Visualizer."""

    def render(self, spec: RenderSpec) -> Open3DHandle:
        self.force_x11_env()

        vis = o3d.visualization.Visualizer()  # type: ignore[reportAttributeAccessIssue]
        vis.create_window(window_name=spec.title)

        geometries: list[o3d.geometry.Geometry] = []
        for layer in spec.layers:
            geom = self.layer_to_geometry(layer)
            if geom is None:
                continue
            if isinstance(geom, list):
                for g in geom:
                    vis.add_geometry(g)
                    geometries.append(g)
            else:
                vis.add_geometry(geom)
                geometries.append(geom)

        vis.poll_events()
        vis.update_renderer()
        return Open3DHandle(vis=vis, geometries=geometries)

    def update(self, handle: Open3DHandle, spec: RenderSpec) -> None:
        for g in handle.geometries:
            handle.vis.remove_geometry(g, reset_bounding_box=False)

        handle.geometries.clear()
        for layer in spec.layers:
            geom = self.layer_to_geometry(layer)
            if geom is None:
                continue
            if isinstance(geom, list):
                for g in geom:
                    handle.vis.add_geometry(g)
                    handle.geometries.append(g)
            else:
                handle.vis.add_geometry(geom)
                handle.geometries.append(geom)

        handle.vis.poll_events()
        handle.vis.update_renderer()

    def layer_to_geometry(self, layer: Any) -> o3d.geometry.Geometry | list[o3d.geometry.Geometry] | None:
        import open3d as o3d

        if isinstance(layer, PointLayer):
            return self.points_to_geometry(layer)
        if isinstance(layer, LineLayer):
            return self.lines_to_geometry(layer)
        if isinstance(layer, Box3DLayer):
            return self.boxes3d_to_geometry(layer)
        if isinstance(layer, TrackLayer):
            return self.tracks_to_geometry(layer)
        return None

    def points_to_geometry(self, layer: PointLayer) -> o3d.geometry.PointCloud:
        import open3d as o3d

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(layer.xyz.astype(float))

        colors = None
        if layer.color is not None:
            colors = layer.color
        elif layer.value is not None and layer.style.colormap:
            colors = self.map_values_to_colors(layer.value, layer.style.colormap)

        if colors is not None:
            if colors.ndim == 1:
                colors = np.repeat(colors[None, :], layer.xyz.shape[0], axis=0)
            pc.colors = o3d.utility.Vector3dVector(colors.astype(float))

        return pc

    def lines_to_geometry(self, layer: LineLayer) -> o3d.geometry.LineSet:
        import open3d as o3d

        segments = layer.segments.astype(float)
        points = segments.reshape(-1, segments.shape[-1])
        lines = np.array([[i, i + 1] for i in range(0, points.shape[0], 2)], dtype=int)

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        return line_set

    def boxes3d_to_geometry(self, layer: Box3DLayer) -> list[o3d.geometry.Geometry]:
        import open3d as o3d

        geoms: list[o3d.geometry.Geometry] = []
        for center, size, yaw in zip(layer.centers, layer.sizes_lwh, layer.yaws):
            box = o3d.geometry.OrientedBoundingBox()
            box.center = center.astype(float)
            box.extent = size.astype(float)
            R = o3d.geometry.get_rotation_matrix_from_xyz([0.0, 0.0, float(yaw)])
            box.R = R
            geoms.append(box)
        return geoms

    def tracks_to_geometry(self, layer: TrackLayer) -> list[o3d.geometry.Geometry]:
        import open3d as o3d

        geoms: list[o3d.geometry.Geometry] = []

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(layer.positions_xyz.astype(float))

        if layer.style.palette is not None:
            colors = [layer.style.palette.get(int(tid), (1.0, 1.0, 1.0)) for tid in layer.track_ids]
            pc.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=float))
        geoms.append(pc)

        if layer.history:
            for trail in layer.history:
                if trail.shape[0] < 2:
                    continue
                segments = np.stack([trail[:-1], trail[1:]], axis=1)
                line_layer = LineLayer(name=f"{layer.name}.trail", meta=layer.meta, segments=segments)
                geoms.append(self.lines_to_geometry(line_layer))

        return geoms

    def map_values_to_colors(self, values: np.ndarray, colormap: str) -> np.ndarray:
        # Simple grayscale fallback; replace with matplotlib if desired.
        _ = colormap
        v = values.astype(float)
        v = (v - v.min()) / (v.max() - v.min() + 1e-6)
        return np.stack([v, v, v], axis=1)

    def force_x11_env(self) -> None:
        # Hyprland/Wayland can break Open3D GLFW windows; force X11/XWayland for stability.
        os.environ.pop("WAYLAND_DISPLAY", None)
        os.environ["XDG_SESSION_TYPE"] = "x11"
