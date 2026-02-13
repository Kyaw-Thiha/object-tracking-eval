from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class _TrackState:
    track_id: int
    box_3d: torch.Tensor
    score: torch.Tensor
    label: torch.Tensor
    velocity_xy: Optional[torch.Tensor]
    last_frame_id: int
    hits: int


class RCBEVDet3DTracker:
    """Minimal velocity-aware 3D tracker for tracking-by-detection."""

    def __init__(
        self,
        center_dist_threshold: float = 2.0,
        max_age: int = 3,
        min_hits: int = 1,
    ):
        self.center_dist_threshold = float(center_dist_threshold)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.reset()

    def reset(self):
        self._tracks: dict[int, _TrackState] = {}
        self._next_track_id = 1

    def _purge_stale(self, frame_id: int):
        stale = [tid for tid, st in self._tracks.items() if frame_id - st.last_frame_id > self.max_age]
        for tid in stale:
            self._tracks.pop(tid, None)

    def _predict_center_xy(self, st: _TrackState, frame_id: int) -> torch.Tensor:
        center_xy = st.box_3d[:2]
        if st.velocity_xy is None:
            return center_xy
        dt = max(frame_id - st.last_frame_id, 0)
        return center_xy + st.velocity_xy * float(dt)

    @torch.no_grad()
    def track(
        self,
        detections_3d: dict[str, torch.Tensor],
        frame_id: int,
    ) -> dict[str, torch.Tensor]:
        boxes = detections_3d["boxes_3d"]
        scores = detections_3d["scores_3d"]
        labels = detections_3d["labels_3d"]
        velocities = detections_3d.get("velocities")

        device = boxes.device
        n = int(boxes.shape[0])

        self._purge_stale(frame_id)

        if n == 0:
            return {
                "boxes_3d": boxes.new_zeros((0, boxes.shape[1] if boxes.ndim == 2 else 0)),
                "scores_3d": scores.new_zeros((0,)),
                "labels_3d": labels.new_zeros((0,), dtype=torch.long),
                "track_ids": torch.zeros((0,), dtype=torch.long, device=device),
                "velocities": boxes.new_zeros((0, 2)),
            }

        track_ids = sorted(self._tracks.keys())
        assigned_det: set[int] = set()
        assigned_track: set[int] = set()
        det_to_track: dict[int, int] = {}

        if track_ids:
            pred_xy = torch.stack([self._predict_center_xy(self._tracks[tid], frame_id) for tid in track_ids], dim=0)
            det_xy = boxes[:, :2]
            dmat = torch.cdist(pred_xy, det_xy, p=2)

            track_labels = torch.stack([self._tracks[tid].label for tid in track_ids]).to(labels.device)
            class_mask = track_labels[:, None] == labels[None, :]
            dmat = torch.where(class_mask, dmat, torch.full_like(dmat, 1e9))

            while True:
                flat_idx = torch.argmin(dmat)
                min_val = dmat.flatten()[flat_idx]
                if min_val > self.center_dist_threshold:
                    break
                t_idx = int(flat_idx // dmat.shape[1])
                d_idx = int(flat_idx % dmat.shape[1])
                if t_idx in assigned_track or d_idx in assigned_det:
                    dmat[t_idx, d_idx] = 1e9
                    continue

                tid = track_ids[t_idx]
                det_to_track[d_idx] = tid
                assigned_track.add(t_idx)
                assigned_det.add(d_idx)
                dmat[t_idx, :] = 1e9
                dmat[:, d_idx] = 1e9

        for d_idx in range(n):
            if d_idx in det_to_track:
                tid = det_to_track[d_idx]
                st = self._tracks[tid]
                st.box_3d = boxes[d_idx]
                st.score = scores[d_idx]
                st.label = labels[d_idx]
                st.velocity_xy = velocities[d_idx] if velocities is not None and velocities.shape[0] > d_idx else None
                st.last_frame_id = frame_id
                st.hits += 1
            else:
                tid = self._next_track_id
                self._next_track_id += 1
                self._tracks[tid] = _TrackState(
                    track_id=tid,
                    box_3d=boxes[d_idx],
                    score=scores[d_idx],
                    label=labels[d_idx],
                    velocity_xy=velocities[d_idx] if velocities is not None and velocities.shape[0] > d_idx else None,
                    last_frame_id=frame_id,
                    hits=1,
                )
                det_to_track[d_idx] = tid

        kept_det_idx = [i for i in range(n) if self._tracks[det_to_track[i]].hits >= self.min_hits]
        if not kept_det_idx:
            return {
                "boxes_3d": boxes.new_zeros((0, boxes.shape[1])),
                "scores_3d": scores.new_zeros((0,)),
                "labels_3d": labels.new_zeros((0,), dtype=torch.long),
                "track_ids": torch.zeros((0,), dtype=torch.long, device=device),
                "velocities": boxes.new_zeros((0, 2)),
            }

        idx = torch.as_tensor(kept_det_idx, dtype=torch.long, device=device)
        out_track_ids = torch.as_tensor([det_to_track[i] for i in kept_det_idx], dtype=torch.long, device=device)

        if velocities is None:
            out_velocities = boxes.new_zeros((len(kept_det_idx), 2))
        else:
            out_velocities = velocities[idx]

        return {
            "boxes_3d": boxes[idx],
            "scores_3d": scores[idx],
            "labels_3d": labels[idx],
            "track_ids": out_track_ids,
            "velocities": out_velocities,
        }
