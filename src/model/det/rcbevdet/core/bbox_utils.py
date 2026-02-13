"""Small bbox utilities used by local CenterPoint head."""

from __future__ import annotations

import torch


def xywhr2xyxyr(boxes_xywhr: torch.Tensor) -> torch.Tensor:
    """Convert rotated boxes from XYWHR to XYXYR."""
    boxes = torch.zeros_like(boxes_xywhr)
    half_w = boxes_xywhr[..., 2] / 2
    half_h = boxes_xywhr[..., 3] / 2

    boxes[..., 0] = boxes_xywhr[..., 0] - half_w
    boxes[..., 1] = boxes_xywhr[..., 1] - half_h
    boxes[..., 2] = boxes_xywhr[..., 0] + half_w
    boxes[..., 3] = boxes_xywhr[..., 1] + half_h
    boxes[..., 4] = boxes_xywhr[..., 4]
    return boxes
