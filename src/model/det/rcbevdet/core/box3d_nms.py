"""Minimal BEV NMS helpers for local CenterPoint head."""

from __future__ import annotations

import numpy as np
import torch
from mmcv.ops import nms_rotated

try:
    import numba
except ImportError:  # pragma: no cover - optional acceleration only.
    class _DummyNumba:
        @staticmethod
        def jit(*args, **kwargs):
            def _wrap(fn):
                return fn
            return _wrap
    numba = _DummyNumba()


@numba.jit(nopython=True)
def oval_nms(dets, thresh_scale, post_max_size=83):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    dx1 = dets[:, 2]
    dy1 = dets[:, 3]
    yaws = dets[:, 4]
    scores = dets[:, -1]
    order = scores.argsort()[::-1].astype(np.int32)
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            dist_x = abs(x1[i] - x1[j])
            dist_y = abs(y1[i] - y1[j])
            dist_x_th = (
                abs(dx1[i] * np.cos(yaws[i]))
                + abs(dx1[j] * np.cos(yaws[j]))
                + abs(dy1[i] * np.sin(yaws[i]))
                + abs(dy1[j] * np.sin(yaws[j]))
            )
            dist_y_th = (
                abs(dx1[i] * np.sin(yaws[i]))
                + abs(dx1[j] * np.sin(yaws[j]))
                + abs(dy1[i] * np.cos(yaws[i]))
                + abs(dy1[j] * np.cos(yaws[j]))
            )
            if dist_x <= dist_x_th * thresh_scale / 2 and dist_y <= dist_y_th * thresh_scale / 2:
                suppressed[j] = 1
    return keep[:post_max_size]


@numba.jit(nopython=True)
def circle_nms(dets, thresh, post_max_size=83):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 2]
    order = scores.argsort()[::-1].astype(np.int32)
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            dist = (x1[i] - x1[j]) ** 2 + (y1[i] - y1[j]) ** 2
            if dist <= thresh:
                suppressed[j] = 1

    if post_max_size < len(keep):
        return keep[:post_max_size]
    return keep


def nms_bev(boxes, scores, thresh, pre_max_size=None, post_max_size=None, xyxyr2xywhr=True):
    assert boxes.size(1) == 5, "Input boxes shape should be [N, 5]"
    order = scores.sort(0, descending=True)[1]
    if pre_max_size is not None:
        order = order[:pre_max_size]
    boxes = boxes[order].contiguous()
    scores = scores[order]

    if xyxyr2xywhr:
        boxes = torch.stack(
            (
                (boxes[:, 0] + boxes[:, 2]) / 2,
                (boxes[:, 1] + boxes[:, 3]) / 2,
                boxes[:, 2] - boxes[:, 0],
                boxes[:, 3] - boxes[:, 1],
                boxes[:, 4],
            ),
            dim=-1,
        )

    keep = nms_rotated(boxes, scores, thresh)[1]
    keep = order[keep]
    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep
