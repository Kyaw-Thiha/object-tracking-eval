"""Shared sequence player utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ...data.schema.frame import Frame
from ..schema.render_spec import RenderSpec

FrameGetter = Callable[[int], Frame]
SpecBuilder = Callable[[Frame], RenderSpec]


@dataclass(frozen=True)
class SequenceRange:
    start: int
    end: int
    step: int = 1

    def indices(self) -> list[int]:
        if self.step <= 0:
            raise ValueError("step must be > 0")
        if self.end < self.start:
            raise ValueError("end must be >= start")
        return list(range(self.start, self.end + 1, self.step))


class BaseSequencePlayer:
    """Base class with shared sequence bookkeeping."""

    def __init__(
        self,
        get_frame: FrameGetter,
        build_spec: SpecBuilder,
        seq: SequenceRange,
        play_interval_s: float = 0.2,
    ) -> None:
        self.get_frame = get_frame
        self.build_spec = build_spec
        self.seq = seq
        self.play_interval_s = play_interval_s

    def indices(self) -> list[int]:
        indices = self.seq.indices()
        if not indices:
            raise ValueError("No indices to render")
        return indices

    def clamp_pos(self, pos: int, indices: list[int]) -> int:
        return max(0, min(pos, len(indices) - 1))
