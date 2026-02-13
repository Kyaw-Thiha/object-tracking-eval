"""Schemas for 3D detection and tracking outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Detection3D:
    """Single-frame 3D detection output."""

    frame_id: int
    video_id: str
    boxes_3d: torch.Tensor
    scores_3d: torch.Tensor
    labels_3d: torch.Tensor
    velocities: Optional[torch.Tensor] = None


@dataclass
class Detection3DBatch:
    """Batch container for 3D detections."""

    detections: list[Detection3D]


@dataclass
class Track3D:
    """Single-frame 3D tracking output."""

    frame_id: int
    video_id: str
    boxes_3d: torch.Tensor
    scores_3d: torch.Tensor
    labels_3d: torch.Tensor
    track_ids: torch.Tensor
    velocities: Optional[torch.Tensor] = None


@dataclass
class Track3DBatch:
    """Batch container for 3D tracking outputs."""

    tracks: list[Track3D]
