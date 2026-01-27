from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from ....sequence_player.base import SequenceRange
from ....sequence_player.napari import NapariSequencePlayer
from ....sequence_player.plotly import PlotlySequencePlayer
from ....backends.napari import NapariBackend
from ....backends.plotly import PlotlyBackend
from ....views.camera import CameraView, CameraViewConfig
from .....data.adapters.camel import CamelAdapter
from .....data.schema.overlay import Box2D, OverlayMeta, OverlaySet
from ....schema.layers import Box2DLayer, RasterLayer
from ....schema.render_spec import RenderSpec
from ....palette import palette_for_dataset


def tint_palette(
    palette: dict[int, tuple[float, float, float]],
    blend: tuple[float, float, float],
    alpha: float,
) -> dict[int, tuple[float, float, float]]:
    tinted: dict[int, tuple[float, float, float]] = {}
    for key, color in palette.items():
        tinted[key] = (
            color[0] * (1 - alpha) + blend[0] * alpha,
            color[1] * (1 - alpha) + blend[1] * alpha,
            color[2] * (1 - alpha) + blend[2] * alpha,
        )
    return tinted


def load_predictions(
    pred_dir: Path,
    adapter: CamelAdapter,
    source_key: str,
    score_threshold: float,
) -> dict[str, dict[int, list[Box2D]]]:
    preds_by_seq: dict[str, dict[int, list[Box2D]]] = {}
    for path in sorted(pred_dir.glob("*.txt")):
        seq_id = path.stem
        preds_by_seq.setdefault(seq_id, {})
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 7:
                    continue
                frame_id = int(float(parts[0]))
                track_id = int(float(parts[1]))
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                score = float(parts[6])
                if score < score_threshold:
                    continue
                class_id = int(float(parts[7])) if len(parts) > 7 else 0
                mapped = adapter.map_category_id(class_id)
                class_id = mapped if mapped is not None else class_id

                xyxy = [x, y, x + w, y + h]
                meta = OverlayMeta(
                    coord_frame="pixel",
                    source=source_key,
                    timestamp=None,
                    sensor_id="cam",
                )
                box = Box2D(
                    meta=meta,
                    xyxy=xyxy,
                    class_id=class_id,
                    confidence=score,
                    track_id=track_id,
                )
                preds_by_seq[seq_id].setdefault(frame_id, []).append(box)
    return preds_by_seq


def ensure_overlay_set(frame) -> OverlaySet:
    if frame.overlays is not None:
        return frame.overlays
    frame.overlays = OverlaySet(
        boxes3D={},
        boxes2D={},
        oriented_boxes2d={},
        radar_dets={},
        radar_polar_dets={},
        tracks={},
    )
    return frame.overlays


def merge_camera_specs(gt_spec: RenderSpec, pred_spec: RenderSpec | None) -> RenderSpec:
    if pred_spec is None:
        return gt_spec

    raster_layers = [layer for layer in gt_spec.layers if isinstance(layer, RasterLayer)]
    gt_layers = [layer for layer in gt_spec.layers if not isinstance(layer, RasterLayer)]
    pred_layers = [layer for layer in pred_spec.layers if not isinstance(layer, RasterLayer)]

    layers = raster_layers + pred_layers + gt_layers
    return RenderSpec(
        title=gt_spec.title,
        coord_frame=gt_spec.coord_frame,
        layers=layers,
        meta=gt_spec.meta,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CAMEL camera view",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Sequence args:
              --end-index <int>   inclusive end index (defaults to last frame)
              --step <int>        stride between frames (default: 1)
              --play-interval <s> seconds per frame when playing (default: 0.2)

            Keybindings:
              Napari: Left/Right arrows step frames
              Plotly: slider + Play/Pause buttons
            """
        ),
    )
    parser.add_argument("--dataset-path", type=str, default="data/camel_dataset")
    parser.add_argument("--ann-file", type=str, default="annotations/half-train_cocoformat.json")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--play-interval", type=float, default=0.2)
    parser.add_argument("--source-key", type=str, default="gt")
    parser.add_argument("--pred-dir", type=str, default=None)
    parser.add_argument("--pred-source-key", type=str, default="pred")
    parser.add_argument("--pred-score-threshold", type=float, default=0.3)
    parser.add_argument("--backend", type=str, choices=["napari", "plotly"], default="napari")
    args = parser.parse_args()

    adapter = CamelAdapter(dataset_path=args.dataset_path, ann_file=args.ann_file, split=args.split)
    view = CameraView()
    end_idx = args.end_index if args.end_index is not None else len(adapter.frames) - 1
    seq = SequenceRange(start=args.index, end=end_idx, step=args.step)

    pred_cache = None
    if args.pred_dir:
        pred_cache = load_predictions(
            pred_dir=Path(args.pred_dir),
            adapter=adapter,
            source_key=args.pred_source_key,
            score_threshold=args.pred_score_threshold,
        )

    base_palette = palette_for_dataset(adapter.dataset_name)
    pred_palette = tint_palette(base_palette, blend=(1.0, 1.0, 1.0), alpha=0.35)

    def get_frame(idx):
        frame = adapter.get_frame(idx)
        if pred_cache and frame.meta and frame.meta.sequence_id:
            seq_preds = pred_cache.get(frame.meta.sequence_id)
            if seq_preds:
                boxes = seq_preds.get(frame.frame_id)
                if boxes:
                    overlays = ensure_overlay_set(frame)
                    overlays.boxes2D[args.pred_source_key] = boxes
        return frame

    def style_pred_layers(spec: RenderSpec) -> RenderSpec:
        for layer in spec.layers:
            if isinstance(layer, Box2DLayer):
                layer.style.palette = pred_palette
                layer.style.line_width = 1.0
        return spec

    def build_spec(frame):
        gt_spec = view.build(frame, CameraViewConfig(sensor_id="cam", source_key=args.source_key))
        if not pred_cache:
            return gt_spec
        pred_spec = view.build(
            frame,
            CameraViewConfig(sensor_id="cam", source_key=args.pred_source_key, show_labels=False, show_tracks=False),
        )
        pred_spec = style_pred_layers(pred_spec)
        return merge_camera_specs(gt_spec, pred_spec)

    if args.backend == "plotly":
        plotly_backend = PlotlyBackend()
        player = PlotlySequencePlayer(
            get_frame=get_frame,
            build_spec=build_spec,
            seq=seq,
            play_interval_s=args.play_interval,
        )

        def build_figure(frame):
            spec = build_spec(frame)
            return plotly_backend.render(spec).fig

        fig = player.build_animation(build_figure)
        fig.show()
    else:
        player = NapariSequencePlayer(
            get_frame=get_frame,
            build_spec=build_spec,
            seq=seq,
            play_interval_s=args.play_interval,
        )
        player.run(NapariBackend())
        import napari

        napari.run()


if __name__ == "__main__":
    main()
