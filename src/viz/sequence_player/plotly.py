"""Plotly sequence player."""

from __future__ import annotations

from typing import Callable
import plotly.graph_objects as go

from .base import BaseSequencePlayer
from ...data.schema.frame import Frame

FigureBuilder = Callable[["Frame"], "go.Figure"]


class PlotlySequencePlayer(BaseSequencePlayer):
    """Plotly animation builder for a sequence."""

    def build_animation(self, build_figure: FigureBuilder):
        indices = self.indices()
        frames: list[go.Frame] = []
        base_fig = None

        for idx in indices:
            frame = self.get_frame(idx)
            fig = build_figure(frame)

            if base_fig is None:
                base_fig = fig

            shapes = getattr(fig.layout, "shapes", None) or []
            frames.append(
                go.Frame(
                    data=fig.data,
                    layout=go.Layout(shapes=shapes),
                    name=str(idx),
                )
            )

        assert base_fig is not None
        base_fig.frames = frames

        steps = []
        for idx in indices:
            steps.append(
                {
                    "label": str(idx),
                    "method": "animate",
                    "args": [
                        [str(idx)],
                        {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
                    ],
                }
            )

        base_fig.update_layout(
            sliders=[{"active": 0, "steps": steps}],
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "fromcurrent": True,
                                    "frame": {"duration": int(self.play_interval_s * 1000), "redraw": True},
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                        },
                    ],
                }
            ],
        )

        return base_fig
