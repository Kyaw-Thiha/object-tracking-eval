from __future__ import annotations

import argparse
import textwrap

from ....sequence_player.base import SequenceRange
from ....sequence_player.napari import NapariSequencePlayer
from ....sequence_player.plotly import PlotlySequencePlayer
from ....backends.napari import NapariBackend
from ....backends.plotly import PlotlyBackend
from ....views.camera import CameraView, CameraViewConfig
from .....data.adapters.camel import CamelAdapter


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
    parser.add_argument("--backend", type=str, choices=["napari", "plotly"], default="napari")
    args = parser.parse_args()

    adapter = CamelAdapter(dataset_path=args.dataset_path, ann_file=args.ann_file, split=args.split)
    view = CameraView()
    end_idx = args.end_index if args.end_index is not None else len(adapter.frames) - 1
    seq = SequenceRange(start=args.index, end=end_idx, step=args.step)

    def build_spec(frame):
        return view.build(frame, CameraViewConfig(sensor_id="cam", source_key=args.source_key))

    if args.backend == "plotly":
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
        player = NapariSequencePlayer(
            get_frame=adapter.get_frame,
            build_spec=build_spec,
            seq=seq,
            play_interval_s=args.play_interval,
        )
        player.run(NapariBackend())
        import napari

        napari.run()


if __name__ == "__main__":
    main()
