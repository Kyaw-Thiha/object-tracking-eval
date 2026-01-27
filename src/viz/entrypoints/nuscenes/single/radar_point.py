from __future__ import annotations

import argparse
import textwrap

from ....backends.plotly import PlotlyBackend
from ....sequence_player.base import SequenceRange
from ....sequence_player.napari import NapariSequencePlayer
from ....sequence_player.plotly import PlotlySequencePlayer
from ....views.radar_point import RadarPointView, RadarPointViewConfig
from ..common import NuscenesArgs, build_backend, load_adapter, resolve_start_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NuScenes radar point view",
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
    parser.add_argument("--dataset-path", type=str, default="data/nuScenes")
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--frame-id", type=int, default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--play-interval", type=float, default=0.2)
    parser.add_argument("--sensor-id", type=str, default="RADAR_FRONT")
    parser.add_argument("--source-key", type=str, default="gt")
    parser.add_argument("--value-key", type=str, default=None)
    parser.add_argument("--units", type=str, default=None)
    parser.add_argument("--show-gt-centers", action="store_true")
    parser.add_argument("--show-gt-footprints", action="store_true")
    parser.add_argument("--use-webgl", action="store_true", help="Enable WebGL scatter for faster playback.")
    parser.add_argument("--backend", type=str, choices=["plotly", "napari"], default="plotly")
    args = parser.parse_args()

    cfg = NuscenesArgs(
        dataset_path=args.dataset_path,
        scene=args.scene,
        frame_id=args.frame_id,
        index=args.index,
        source_key=args.source_key,
        sensor_id=args.sensor_id,
        backend=args.backend,
    )

    adapter = load_adapter(cfg.dataset_path)
    view = RadarPointView()
    start_idx = resolve_start_index(args, adapter)
    end_idx = args.end_index if args.end_index is not None else len(adapter.frames) - 1
    seq = SequenceRange(start=start_idx, end=end_idx, step=args.step)

    def build_spec(frame):
        return view.build(
            frame,
            RadarPointViewConfig(
                sensor_id=cfg.sensor_id or "",
                source_key=cfg.source_key,
                value_key=args.value_key,
                units=args.units,
                show_gt_centers=args.show_gt_centers,
                show_gt_footprints=args.show_gt_footprints,
            ),
        )

    if cfg.backend == "plotly":
        use_webgl = args.use_webgl and not (args.show_gt_centers or args.show_gt_footprints)
        plotly_backend = PlotlyBackend(use_webgl=use_webgl)
        player = PlotlySequencePlayer(
            get_frame=adapter.get_frame,
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
        backend = build_backend(cfg.backend)
        player = NapariSequencePlayer(
            get_frame=adapter.get_frame,
            build_spec=build_spec,
            seq=seq,
            play_interval_s=args.play_interval,
        )
        player.run(backend)
        import napari

        napari.run()


if __name__ == "__main__":
    main()
