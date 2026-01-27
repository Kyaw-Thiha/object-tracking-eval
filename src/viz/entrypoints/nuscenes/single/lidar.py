from __future__ import annotations

import argparse
import textwrap

from ....backends.plotly import PlotlyBackend
from ....sequence_player.base import SequenceRange
from ....sequence_player.open3d import Open3DSequencePlayer
from ....sequence_player.plotly import PlotlySequencePlayer
from ....views.lidar import LidarView, LidarViewConfig
from ..common import NuscenesArgs, build_backend, load_adapter, resolve_start_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NuScenes lidar view",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Sequence args:
              --end-index <int>   inclusive end index (defaults to last frame)
              --step <int>        stride between frames (default: 1)
              --play-interval <s> seconds per frame when playing (default: 0.2)

            Keybindings:
              Open3D: A/D step frames, Space toggles play
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
    parser.add_argument("--sensor-id", type=str, default="LIDAR_TOP")
    parser.add_argument("--source-key", type=str, default="gt")
    parser.add_argument("--backend", type=str, choices=["open3d", "plotly"], default="open3d")
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
    view = LidarView()
    start_idx = resolve_start_index(args, adapter)
    end_idx = args.end_index if args.end_index is not None else len(adapter.frames) - 1
    seq = SequenceRange(start=start_idx, end=end_idx, step=args.step)

    def build_spec(frame):
        return view.build(frame, LidarViewConfig(sensor_id=cfg.sensor_id or "", source_key=cfg.source_key))

    if cfg.backend == "plotly":
        plotly_backend = PlotlyBackend()
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
        player = Open3DSequencePlayer(
            get_frame=adapter.get_frame,
            build_spec=build_spec,
            seq=seq,
            play_interval_s=args.play_interval,
        )
        player.run(backend)


if __name__ == "__main__":
    main()
