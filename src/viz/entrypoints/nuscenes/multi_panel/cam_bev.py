from __future__ import annotations

import argparse

from ....views.bev import BEVView, BEVViewConfig
from ....views.camera import CameraView, CameraViewConfig
from ..common import NuscenesArgs, load_adapter, render_plotly_grid, resolve_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="NuScenes camera + BEV multi-panel")
    parser.add_argument("--dataset-path", type=str, default="data/nuScenes")
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--frame-id", type=int, default=None)
    parser.add_argument("--index", type=int, default=None)
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
    frame = resolve_frame(cfg, adapter)

    cam_spec = CameraView().build(frame, CameraViewConfig(sensor_id=args.camera_id, source_key=args.source_key))

    sensor_ids = [s.strip() for s in args.sensor_ids.split(",") if s.strip()]
    bev_spec = BEVView().build(frame, BEVViewConfig(sensor_ids=sensor_ids, source_keys=[args.source_key]))

    handle = render_plotly_grid(specs=[cam_spec, bev_spec], titles=["Camera", "BEV"], rows=1, cols=2)
    handle.fig.show()


if __name__ == "__main__":
    main()
