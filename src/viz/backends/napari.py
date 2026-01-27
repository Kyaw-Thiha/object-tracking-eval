"""Napari backend for RenderSpec visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import os

import numpy as np
from napari import Viewer

from .base import BaseBackend
from ..schema.render_spec import RenderSpec
from ..schema.layers import RasterLayer, PointLayer, Box2DLayer, TextLayer, TrackLayer


@dataclass
class NapariHandle:
    viewer: Viewer


class NapariBackend(BaseBackend):
    """Render 2D layers into a napari Viewer."""

    def render(self, spec: RenderSpec) -> NapariHandle:
        self.configure_qt_env()

        viewer = Viewer(title=spec.title)
        for layer in spec.layers:
            self.add_layer(viewer, layer)
        return NapariHandle(viewer=viewer)

    def update(self, handle: NapariHandle, spec: RenderSpec) -> None:
        viewer = handle.viewer
        for layer in list(viewer.layers):
            viewer.layers.remove(layer)
        for layer in spec.layers:
            self.add_layer(viewer, layer)

    def add_layer(self, viewer: Viewer, layer: Any) -> None:
        if isinstance(layer, RasterLayer):
            viewer.add_image(layer.data, name=layer.name)

        elif isinstance(layer, PointLayer):
            pts = layer.xyz[:, :2]
            viewer.add_points(pts, name=layer.name, size=int(layer.style.point_size))

        elif isinstance(layer, Box2DLayer):
            x1, y1, x2, y2 = layer.xyxy[:, 0], layer.xyxy[:, 1], layer.xyxy[:, 2], layer.xyxy[:, 3]
            if layer.meta.coord_frame == "pixel":
                verts = np.stack(
                    [
                        np.stack([y1, x1], axis=1),
                        np.stack([y1, x2], axis=1),
                        np.stack([y2, x2], axis=1),
                        np.stack([y2, x1], axis=1),
                    ],
                    axis=1,
                )
            else:
                verts = np.stack(
                    [
                        np.stack([x1, y1], axis=1),
                        np.stack([x2, y1], axis=1),
                        np.stack([x2, y2], axis=1),
                        np.stack([x1, y2], axis=1),
                    ],
                    axis=1,
                )
            viewer.add_shapes(verts, shape_type="polygon", name=layer.name, edge_width=int(layer.style.line_width))

        elif isinstance(layer, TextLayer):
            xy = layer.xy[:, :2]
            if layer.meta.coord_frame == "pixel":
                xy = xy[:, [1, 0]]
            viewer.add_text(xy, layer.texts, name=layer.name)

        elif isinstance(layer, TrackLayer):
            pts = layer.positions_xyz[:, :2]
            viewer.add_points(pts, name=layer.name, size=int(layer.style.point_size))

    def configure_qt_env(self) -> None:
        # Hyprland/Wayland can be flaky for Qt; allow Wayland with X11 fallback.
        os.environ.setdefault("QT_QPA_PLATFORM", "wayland;xcb")
