from __future__ import annotations

import argparse

from ....views.radar_point import RadarPointView, RadarPointViewConfig
from ..common import NuscenesArgs, build_backend, load_adapter, resolve_frame, show_backend


def main() -> None:
    parser = argparse.ArgumentParser(description="NuScenes radar point view")
    parser.add_argument("--dataset-path", type=str, default="data/nuScenes")
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--frame-id", type=int, default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--sensor-id", type=str, default="RADAR_FRONT")
    parser.add_argument("--source-key", type=str, default="gt")
    parser.add_argument("--value-key", type=str, default=None)
    parser.add_argument("--units", type=str, default=None)
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
    frame = resolve_frame(cfg, adapter)
    view = RadarPointView()
    spec = view.build(
        frame,
        RadarPointViewConfig(
            sensor_id=cfg.sensor_id or "",
            source_key=cfg.source_key,
            value_key=args.value_key,
            units=args.units,
        ),
    )
    backend = build_backend(cfg.backend)
    handle = backend.render(spec)
    show_backend(handle, cfg.backend)


if __name__ == "__main__":
    main()
