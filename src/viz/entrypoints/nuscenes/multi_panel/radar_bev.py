from __future__ import annotations

import argparse
import textwrap

from ....sequence_player.base import SequenceRange
from ....sequence_player.plotly import PlotlySequencePlayer
from ....views.bev import BEVView, BEVViewConfig
from ....views.radar_grid import RadarGridView, RadarGridViewConfig
from ..common import NuscenesArgs, load_adapter, render_plotly_grid, resolve_start_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NuScenes radar + BEV multi-panel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Sequence args:
              --end-index <int>   inclusive end index (defaults to last frame)
              --step <int>        stride between frames (default: 1)
              --play-interval <s> seconds per frame when playing (default: 0.2)

            Keybindings:
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
    parser.add_argument("--grid-name", type=str, default="RA")
    parser.add_argument("--display", type=str, choices=["pixel", "polar"], default="pixel")
    parser.add_argument("--source-key", type=str, default="gt")
    parser.add_argument("--bev-max-points", type=int, default=None)
    args = parser.parse_args()

    cfg = NuscenesArgs(
        dataset_path=args.dataset_path,
        scene=args.scene,
        frame_id=args.frame_id,
        index=args.index,
        source_key=args.source_key,
        sensor_id=args.sensor_id,
        backend="plotly",
    )

    adapter = load_adapter(cfg.dataset_path, synthesize_radar_grids=True)
    radar_view = RadarGridView()
    bev_view = BEVView()

    start_idx = resolve_start_index(args, adapter)
    end_idx = args.end_index if args.end_index is not None else len(adapter.frames) - 1
    seq = SequenceRange(start=start_idx, end=end_idx, step=args.step)

    def build_spec(frame):
        return radar_view.build(
            frame,
            RadarGridViewConfig(
                sensor_id=args.sensor_id,
                source_key=args.source_key,
                grid_name=args.grid_name,
                display=args.display,
            ),
        )

    def build_figure(frame):
        radar_spec = radar_view.build(
            frame,
            RadarGridViewConfig(
                sensor_id=args.sensor_id,
                source_key=args.source_key,
                grid_name=args.grid_name,
                display=args.display,
            ),
        )
        bev_spec = bev_view.build(
            frame,
            BEVViewConfig(
                sensor_ids=["LIDAR_TOP", args.sensor_id],
                source_keys=[args.source_key],
                max_points=args.bev_max_points,
            ),
        )
        return render_plotly_grid(
            specs=[radar_spec, bev_spec],
            titles=["Radar Grid", "BEV"],
            rows=1,
            cols=2,
            use_webgl=False,
        ).fig

    player = PlotlySequencePlayer(
        get_frame=adapter.get_frame,
        build_spec=build_spec,
        seq=seq,
        play_interval_s=args.play_interval,
    )
    fig = player.build_animation(build_figure)
    fig.show()


if __name__ == "__main__":
    main()
