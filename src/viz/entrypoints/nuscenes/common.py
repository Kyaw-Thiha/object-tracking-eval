from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ....data.adapters.nuscenes import NuScenesAdapter
from ....data.schema.frame import Frame
from ...backends.base import BaseBackend
from ...backends.napari import NapariBackend
from ...backends.open3d import Open3DBackend
from ...backends.plotly import PlotlyBackend, PlotlyHandle
from ...schema.render_spec import RenderSpec

BackendName = Literal["open3d", "napari", "plotly"]


@dataclass
class NuscenesArgs:
    dataset_path: str
    scene: str | None
    frame_id: int | None
    index: int | None
    source_key: str
    sensor_id: str | None
    backend: BackendName


def load_adapter(dataset_path: str, synthesize_radar_grids: bool = False) -> NuScenesAdapter:
    return NuScenesAdapter(dataset_path=dataset_path, synthesize_radar_grids=synthesize_radar_grids)


def get_frame_by_scene(adapter: NuScenesAdapter, scene: str, frame_id: int) -> Frame:
    for idx, info in enumerate(adapter.frames):
        meta = info.get("meta")
        if not meta:
            continue
        if meta.sequence_id == scene and info["frame_id"] == frame_id:
            return adapter.get_frame(idx)
    raise ValueError(f"Frame not found for scene={scene} frame_id={frame_id}")


def get_frame_index_by_scene(adapter: NuScenesAdapter, scene: str, frame_id: int) -> int:
    for idx, info in enumerate(adapter.frames):
        meta = info.get("meta")
        if not meta:
            continue
        if meta.sequence_id == scene and info["frame_id"] == frame_id:
            return idx
    raise ValueError(f"Frame not found for scene={scene} frame_id={frame_id}")


def get_frame_by_index(adapter: NuScenesAdapter, index: int) -> Frame:
    return adapter.get_frame(index)


def resolve_frame(args: NuscenesArgs, adapter: NuScenesAdapter) -> Frame:
    if args.index is not None:
        return get_frame_by_index(adapter, args.index)
    if args.scene is not None and args.frame_id is not None:
        return get_frame_by_scene(adapter, args.scene, args.frame_id)
    raise ValueError("Provide either --index or both --scene and --frame-id")


def resolve_start_index(args, adapter: NuScenesAdapter) -> int:
    if getattr(args, "index", None) is not None:
        return args.index
    if getattr(args, "scene", None) is not None and getattr(args, "frame_id", None) is not None:
        return get_frame_index_by_scene(adapter, args.scene, args.frame_id)
    return 0


def build_backend(name: BackendName) -> BaseBackend:
    if name == "open3d":
        return Open3DBackend()
    if name == "napari":
        return NapariBackend()
    if name == "plotly":
        return PlotlyBackend()
    raise ValueError(f"Unknown backend: {name}")


def show_backend(handle, backend: BackendName) -> None:
    if backend == "open3d":
        handle.vis.run()
    elif backend == "napari":
        import napari

        napari.run()
    elif backend == "plotly":
        handle.fig.show()


def render_plotly_grid(
    specs: Sequence[RenderSpec],
    titles: Sequence[str],
    rows: int,
    cols: int,
    use_webgl: bool = True,
) -> PlotlyHandle:
    if len(specs) != rows * cols:
        raise ValueError("specs length must match rows * cols")

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles)
    backend = PlotlyBackend(use_webgl=use_webgl)

    for idx, spec in enumerate(specs):
        row = idx // cols + 1
        col = idx % cols + 1

        handle = backend.render(spec)
        for trace in handle.fig.data:
            fig.add_trace(trace, row=row, col=col)

        for shape in handle.fig.layout.shapes or []:
            fig.add_shape(shape, row=row, col=col)

    fig.update_layout(title=" / ".join(titles), showlegend=False)
    return PlotlyHandle(fig=fig)
