"""Napari sequence player."""

from __future__ import annotations

from napari import Viewer

from viz.backends.napari import NapariBackend

from .base import BaseSequencePlayer


class NapariSequencePlayer(BaseSequencePlayer):
    """Napari key-controlled sequence player."""

    def run(self, backend: NapariBackend) -> None:
        indices = self.indices()

        frame = self.get_frame(indices[0])
        spec = self.build_spec(frame)
        handle = backend.render(spec)
        viewer: Viewer = handle.viewer

        state = {"pos": 0}

        def goto_pos(pos: int) -> None:
            pos = self.clamp_pos(pos, indices)
            state["pos"] = pos
            idx = indices[pos]
            frame = self.get_frame(idx)
            spec = self.build_spec(frame)
            backend.update(handle, spec)

        @viewer.bind_key("Left")
        def _prev(_viewer) -> None:
            goto_pos(state["pos"] - 1)

        @viewer.bind_key("Right")
        def _next(_viewer) -> None:
            goto_pos(state["pos"] + 1)
