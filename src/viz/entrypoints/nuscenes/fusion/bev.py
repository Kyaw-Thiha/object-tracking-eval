from __future__ import annotations

import argparse

from ....views.bev import BEVView, BEVViewConfig
from ..common import NuscenesArgs, build_backend, load_adapter, resolve_frame, show_backend


def main() -> None:
    parser = argparse.ArgumentParser(description="NuScenes BEV fusion view")
    parser.add_argument("--dataset-path", type=str, default="data/nuScenes")
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--frame-id", type=int, default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--sensor-ids", type=str, default="LIDAR_TOP,RADAR_FRONT")
    parser.add_argument("--source-keys", type=str, default="gt")
    parser.add_argument("--backend", type=str, choices=["open3d", "plotly"], default="open3d")
    args = parser.parse_args()

    cfg = NuscenesArgs(
        dataset_path=args.dataset_path,
        scene=args.scene,
        frame_id=args.frame_id,
        index=args.index,
        source_key="gt",
        sensor_id=None,
        backend=args.backend,
    )

    sensor_ids = [s.strip() for s in args.sensor_ids.split(",") if s.strip()]
    source_keys = [s.strip() for s in args.source_keys.split(",") if s.strip()]

    adapter = load_adapter(cfg.dataset_path)
    frame = resolve_frame(cfg, adapter)
    view = BEVView()
    spec = view.build(frame, BEVViewConfig(sensor_ids=sensor_ids, source_keys=source_keys))
    backend = build_backend(cfg.backend)
    handle = backend.render(spec)
    show_backend(handle, cfg.backend)


if __name__ == "__main__":
    main()
