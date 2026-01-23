from __future__ import annotations

import argparse

from ....views.bev import BEVView, BEVViewConfig
from ....views.radar_grid import RadarGridView, RadarGridViewConfig
from ..common import NuscenesArgs, load_adapter, render_plotly_grid, resolve_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="NuScenes radar + BEV multi-panel")
    parser.add_argument("--dataset-path", type=str, default="data/nuScenes")
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--frame-id", type=int, default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--sensor-id", type=str, default="RADAR_FRONT")
    parser.add_argument("--grid-name", type=str, default="RA")
    parser.add_argument("--display", type=str, choices=["pixel", "polar"], default="pixel")
    parser.add_argument("--source-key", type=str, default="gt")
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

    adapter = load_adapter(cfg.dataset_path)
    frame = resolve_frame(cfg, adapter)

    radar_spec = RadarGridView().build(
        frame,
        RadarGridViewConfig(
            sensor_id=args.sensor_id,
            source_key=args.source_key,
            grid_name=args.grid_name,
            display=args.display,
        ),
    )

    bev_spec = BEVView().build(
        frame,
        BEVViewConfig(sensor_ids=["LIDAR_TOP", args.sensor_id], source_keys=[args.source_key]),
    )

    handle = render_plotly_grid(specs=[radar_spec, bev_spec], titles=["Radar Grid", "BEV"], rows=1, cols=2)
    handle.fig.show()


if __name__ == "__main__":
    main()
