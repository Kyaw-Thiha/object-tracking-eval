from __future__ import annotations

import argparse

from ....sequence_player.base import SequenceRange
from ....sequence_player.plotly import PlotlySequencePlayer
from ....views.bev import BEVView, BEVViewConfig
from ....views.camera import CameraView, CameraViewConfig
from ..common import NuscenesArgs, load_adapter, render_plotly_grid, resolve_start_index


def main() -> None:
    parser = argparse.ArgumentParser(description="NuScenes camera + BEV multi-panel")
    parser.add_argument("--dataset-path", type=str, default="data/nuScenes")
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--frame-id", type=int, default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--play-interval", type=float, default=0.2)
    parser.add_argument("--camera-id", type=str, default="CAM_FRONT")
    parser.add_argument("--sensor-ids", type=str, default="LIDAR_TOP,RADAR_FRONT")
    parser.add_argument("--source-key", type=str, default="gt")
    args = parser.parse_args()

    cfg = NuscenesArgs(
        dataset_path=args.dataset_path,
        scene=args.scene,
        frame_id=args.frame_id,
        index=args.index,
        source_key=args.source_key,
        sensor_id=args.camera_id,
        backend="plotly",
    )

    adapter = load_adapter(cfg.dataset_path)
    sensor_ids = [s.strip() for s in args.sensor_ids.split(",") if s.strip()]
    cam_view = CameraView()
    bev_view = BEVView()

    start_idx = resolve_start_index(args, adapter)
    end_idx = args.end_index if args.end_index is not None else len(adapter.frames) - 1
    seq = SequenceRange(start=start_idx, end=end_idx, step=args.step)

    def build_spec(frame):
        return cam_view.build(frame, CameraViewConfig(sensor_id=args.camera_id, source_key=args.source_key))

    def build_figure(frame):
        cam_spec = cam_view.build(frame, CameraViewConfig(sensor_id=args.camera_id, source_key=args.source_key))
        bev_spec = bev_view.build(frame, BEVViewConfig(sensor_ids=sensor_ids, source_keys=[args.source_key]))
        return render_plotly_grid(specs=[cam_spec, bev_spec], titles=["Camera", "BEV"], rows=1, cols=2).fig

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
