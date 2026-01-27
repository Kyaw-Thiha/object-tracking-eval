"""Plotly backend for RenderSpec visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go

from .base import BaseBackend
from ..schema.render_spec import RenderSpec
from ..schema.layers import RasterLayer, PointLayer, Box2DLayer, Box3DLayer, TextLayer, TrackLayer
from ..geometry import box3d_bev_corners
from ..palette import DEFAULT_COLOR


@dataclass
class PlotlyHandle:
    fig: go.Figure


class PlotlyBackend(BaseBackend):
    """Render 2D layers into a Plotly Figure."""

    def __init__(self, use_webgl: bool = True) -> None:
        self.use_webgl = use_webgl

    def render(self, spec: RenderSpec) -> PlotlyHandle:
        fig = go.Figure()
        fig.update_layout(title=spec.title, yaxis=dict(autorange="reversed"))

        for layer in spec.layers:
            self.add_layer(fig, layer)

        self.add_layer_filters(fig, spec)
        return PlotlyHandle(fig=fig)

    def update(self, handle: PlotlyHandle, spec: RenderSpec) -> None:
        fig = handle.fig
        fig.data = ()
        fig.update_layout(shapes=[], updatemenus=[])
        fig.update_layout(title=spec.title, yaxis=dict(autorange="reversed"))

        for layer in spec.layers:
            self.add_layer(fig, layer)

        self.add_layer_filters(fig, spec)

    def add_layer(self, fig: go.Figure, layer: Any) -> None:
        def rgb_string(color: tuple[float, float, float]) -> str:
            return "rgb(%d,%d,%d)" % tuple((np.array(color) * 255).astype(int))

        def rgba_string(color: tuple[float, float, float], alpha: float) -> str:
            vals = tuple((np.array(color) * 255).astype(int).tolist())
            return f"rgba({vals[0]},{vals[1]},{vals[2]},{alpha:.3f})"

        def color_for_class(layer_obj: Any, class_id: int | None) -> tuple[float, float, float]:
            if class_id is None:
                return DEFAULT_COLOR
            palette = getattr(layer_obj.style, "palette", None)
            if palette is None:
                return DEFAULT_COLOR
            return palette.get(int(class_id), DEFAULT_COLOR)

        if isinstance(layer, RasterLayer):
            if layer.display == "polar" and layer.axes and layer.bins:
                self.add_raster_polar(fig, layer)
            elif layer.axes and layer.bins:
                x = layer.bins[layer.axes[1]]
                y = layer.bins[layer.axes[0]]
                fig.add_trace(
                    go.Heatmap(
                        x=x,
                        y=y,
                        z=layer.data,
                        colorscale=layer.style.colormap or "Viridis",
                        showscale=False,
                        name=layer.name,
                    )
                )
            else:
                fig.add_trace(go.Image(z=layer.data, name=layer.name))

        elif isinstance(layer, PointLayer):
            marker: dict[str, Any] = {"size": layer.style.point_size}
            if layer.color is not None:
                if layer.color.ndim == 1:
                    marker["color"] = layer.color
                else:
                    marker["color"] = ["rgb(%d,%d,%d)" % tuple((c * 255).astype(int)) for c in layer.color]
            elif layer.value is not None:
                marker["color"] = layer.value
                marker["colorscale"] = layer.style.colormap or "Viridis"

            scatter_cls = go.Scattergl if self.use_webgl else go.Scatter
            fig.add_trace(
                scatter_cls(
                    x=layer.xyz[:, 0],
                    y=layer.xyz[:, 1],
                    mode="markers",
                    name=layer.name,
                    marker=marker,
                )
            )

        elif isinstance(layer, Box2DLayer):
            class_ids = layer.class_ids if layer.class_ids is not None else [None] * len(layer.xyxy)
            for box, class_id in zip(layer.xyxy, class_ids):
                x1, y1, x2, y2 = box.tolist()
                color = color_for_class(layer, class_id)
                fig.add_shape(
                    type="rect",
                    x0=x1,
                    y0=y1,
                    x1=x2,
                    y1=y2,
                    line=dict(width=layer.style.line_width, color=rgb_string(color)),
                    fillcolor=rgba_string(color, 0.15),
                )

        elif isinstance(layer, Box3DLayer):
            class_ids = layer.class_ids if layer.class_ids is not None else [None] * len(layer.centers)
            for center, size, yaw, class_id in zip(layer.centers, layer.sizes_lwh, layer.yaws, class_ids):
                xy = box3d_bev_corners(center, size, yaw)
                color = color_for_class(layer, class_id)
                fig.add_trace(
                    go.Scatter(
                        x=xy[:, 0],
                        y=xy[:, 1],
                        mode="lines",
                        name=layer.name,
                        showlegend=False,
                        line=dict(width=layer.style.line_width, color=rgb_string(color)),
                        fill="toself",
                        fillcolor=rgba_string(color, 0.15),
                    )
                )

        elif isinstance(layer, TextLayer):
            fig.add_trace(
                go.Scatter(
                    x=layer.xy[:, 0],
                    y=layer.xy[:, 1],
                    mode="text",
                    text=layer.texts,
                    name=layer.name,
                )
            )

        elif isinstance(layer, TrackLayer):
            marker: dict[str, Any] = {"size": layer.style.point_size}
            if layer.style.palette is not None:
                marker["color"] = [layer.style.palette.get(int(tid), (1.0, 1.0, 1.0)) for tid in layer.track_ids]

            scatter_cls = go.Scattergl if self.use_webgl else go.Scatter
            fig.add_trace(
                scatter_cls(
                    x=layer.positions_xyz[:, 0],
                    y=layer.positions_xyz[:, 1],
                    mode="markers",
                    name=layer.name,
                    marker=marker,
                )
            )

    def add_raster_polar(self, fig: go.Figure, layer: RasterLayer) -> None:
        if layer.axes is None or layer.bins is None:
            return
        r = layer.bins[layer.axes[0]]
        theta = layer.bins[layer.axes[1]]
        R, T = np.meshgrid(r, theta, indexing="ij")
        fig.add_trace(
            go.Scatterpolar(
                r=R.flatten(),
                theta=np.degrees(T.flatten()),
                mode="markers",
                marker=dict(color=layer.data.flatten(), colorscale=layer.style.colormap or "Viridis", size=2),
                name=layer.name,
            )
        )

    def add_layer_filters(self, fig: go.Figure, spec: RenderSpec) -> None:
        sources = sorted({layer.meta.source for layer in spec.layers})
        kinds = sorted({layer.meta.kind for layer in spec.layers})

        buttons = []
        for src in sources:
            vis = [layer.meta.source == src for layer in spec.layers]
            buttons.append(dict(label=f"source:{src}", method="update", args=[{"visible": vis}]))

        for kind in kinds:
            vis = [layer.meta.kind == kind for layer in spec.layers]
            buttons.append(dict(label=f"kind:{kind}", method="update", args=[{"visible": vis}]))

        if buttons:
            fig.update_layout(updatemenus=[dict(type="dropdown", buttons=buttons, x=1.02, y=1.0)])
