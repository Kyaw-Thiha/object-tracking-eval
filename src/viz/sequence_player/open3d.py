"""Open3D sequence player."""

from __future__ import annotations

import time

from viz.backends.open3d import Open3DBackend

from .base import BaseSequencePlayer


class Open3DSequencePlayer(BaseSequencePlayer):
    """Open3D key-controlled sequence player."""

    def run(self, backend: Open3DBackend) -> None:
        indices = self.indices()

        frame = self.get_frame(indices[0])
        spec = self.build_spec(frame)
        handle = backend.render(spec)
        vis = handle.vis

        state = {"pos": 0, "playing": False, "last": time.time()}

        def goto_pos(pos: int) -> None:
            pos = self.clamp_pos(pos, indices)
            state["pos"] = pos

            idx = indices[pos]
            frame = self.get_frame(idx)
            spec = self.build_spec(frame)

            backend.update(handle, spec)
            vis.poll_events()
            vis.update_renderer()

        def on_prev(_vis) -> bool:
            goto_pos(state["pos"] - 1)
            return False

        def on_next(_vis) -> bool:
            goto_pos(state["pos"] + 1)
            return False

        def on_toggle(_vis) -> bool:
            state["playing"] = not state["playing"]
            state["last"] = time.time()
            return False

        def on_anim(_vis) -> bool:
            if not state["playing"]:
                return False
            now = time.time()
            if now - state["last"] >= self.play_interval_s:
                state["last"] = now
                if state["pos"] + 1 >= len(indices):
                    state["playing"] = False
                else:
                    goto_pos(state["pos"] + 1)
            return False

        vis.register_key_callback(ord("A"), on_prev)
        vis.register_key_callback(ord("D"), on_next)
        vis.register_key_callback(ord(" "), on_toggle)
        vis.register_animation_callback(on_anim)
        vis.run()
