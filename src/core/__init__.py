"""Core public API with minimal side effects.

Keep exports explicit so importing `core` does not pull legacy MMTrack dataset,
inference, or evaluation modules unintentionally.
"""

from .utils import outs2results, results2outs
from .visualization import (
    get_ellipse_params,
    imshow_det_bboxes,
    imshow_mot_errors,
    show_track_result,
)

__all__ = [
    "outs2results",
    "results2outs",
    "get_ellipse_params",
    "imshow_det_bboxes",
    "imshow_mot_errors",
    "show_track_result",
]
