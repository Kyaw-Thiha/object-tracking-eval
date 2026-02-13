from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Callable, cast

import torch
import torch.nn as nn
from mmcv import Config

# Ensure local RCBEVDet modules register into local registry.
import model.det.rcbevdet as _rcbevdet_registration
from model.det.rcbevdet import build_detector

SRC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SRC_ROOT.parent
# Keep an explicit reference so static analyzers treat side-effect import as used.
_ = _rcbevdet_registration


def _unwrap_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    cleaned = {}
    for k, v in state_dict.items():
        key = k
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = v
    return cleaned


def _boxes3d_to_tensor(boxes_3d: Any, device: torch.device) -> torch.Tensor:
    if isinstance(boxes_3d, torch.Tensor):
        return boxes_3d.to(device=device, dtype=torch.float32)
    if hasattr(boxes_3d, "tensor"):
        return boxes_3d.tensor.to(device=device, dtype=torch.float32)
    raise TypeError(f"Unsupported boxes_3d type: {type(boxes_3d)}")


def _yaw_to_rotmat(yaw: torch.Tensor) -> torch.Tensor:
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    z = torch.zeros_like(c)
    o = torch.ones_like(c)
    rot = torch.stack(
        [
            torch.stack([c, -s, z], dim=-1),
            torch.stack([s, c, z], dim=-1),
            torch.stack([z, z, o], dim=-1),
        ],
        dim=-2,
    )
    return rot


def _boxes3d_corners(boxes_xyzlwhyaw: torch.Tensor) -> torch.Tensor:
    # boxes: [N, >=7] in ego frame, size order [l, w, h].
    center = boxes_xyzlwhyaw[:, 0:3]
    lwh = boxes_xyzlwhyaw[:, 3:6]
    yaw = boxes_xyzlwhyaw[:, 6]

    l2 = lwh[:, 0] * 0.5
    w2 = lwh[:, 1] * 0.5
    h2 = lwh[:, 2] * 0.5

    corners_local = torch.stack(
        [
            torch.stack([l2, w2, h2], dim=-1),
            torch.stack([l2, -w2, h2], dim=-1),
            torch.stack([-l2, -w2, h2], dim=-1),
            torch.stack([-l2, w2, h2], dim=-1),
            torch.stack([l2, w2, -h2], dim=-1),
            torch.stack([l2, -w2, -h2], dim=-1),
            torch.stack([-l2, -w2, -h2], dim=-1),
            torch.stack([-l2, w2, -h2], dim=-1),
        ],
        dim=1,
    )

    rot = _yaw_to_rotmat(yaw)
    corners = torch.einsum("nij,nkj->nki", rot, corners_local) + center[:, None, :]
    return corners


class RCBEVDetWrapper(nn.Module):
    def __init__(
        self,
        detector: nn.Module,
        classes: list[str],
        device: str,
        project_to_2d_fallback: bool = True,
    ):
        super().__init__()
        self.detector = detector
        self.device = torch.device(device)
        self.detector.to(self.device)
        self.detector.eval()
        self.classes = classes
        self.project_to_2d_fallback = project_to_2d_fallback

    def get_classes(self) -> list[str]:
        return self.classes

    @staticmethod
    def _require_targets(targets: Any, batch_size: int) -> list[dict[str, Any]]:
        if not isinstance(targets, list) or len(targets) != batch_size:
            raise ValueError("RCBEVDet requires targets as list[dict] aligned with batch size.")
        required = ["img_inputs", "radar_points", "img_metas"]
        for i, t in enumerate(targets):
            for key in required:
                if key not in t:
                    raise KeyError(f"targets[{i}] missing required key '{key}'")
        return targets

    @staticmethod
    def _stack_temporal_camera_inputs(targets: list[dict[str, Any]], num_frame: int):
        imgs = torch.stack([t["img_inputs"]["imgs"] for t in targets], dim=0)
        sensor2egos = torch.stack([t["img_inputs"]["sensor2egos"] for t in targets], dim=0)
        ego2globals = torch.stack([t["img_inputs"]["ego2globals"] for t in targets], dim=0)
        intrins = torch.stack([t["img_inputs"]["intrins"] for t in targets], dim=0)
        post_rots = torch.stack([t["img_inputs"]["post_rots"] for t in targets], dim=0)
        post_trans = torch.stack([t["img_inputs"]["post_trans"] for t in targets], dim=0)
        bda = torch.stack([t["img_inputs"]["bda"] for t in targets], dim=0)

        if num_frame > 1 and imgs.shape[1] % num_frame != 0:
            # Producer currently provides current frame cameras only. Repeat per
            # camera to satisfy BEVDet4D tensor shape [B, num_cams*num_frame, ...].
            b, n, c, h, w = imgs.shape
            imgs = imgs.unsqueeze(2).repeat(1, 1, num_frame, 1, 1, 1).reshape(b, n * num_frame, c, h, w)

            def _repeat_pose(x: torch.Tensor) -> torch.Tensor:
                b1, n1 = x.shape[:2]
                tail = x.shape[2:]
                return x.unsqueeze(2).repeat(1, 1, num_frame, *([1] * len(tail))).reshape(b1, n1 * num_frame, *tail)

            sensor2egos = _repeat_pose(sensor2egos)
            ego2globals = _repeat_pose(ego2globals)
            intrins = _repeat_pose(intrins)
            post_rots = _repeat_pose(post_rots)
            post_trans = _repeat_pose(post_trans)

        return [
            imgs,
            sensor2egos,
            ego2globals,
            intrins,
            post_rots,
            post_trans,
            bda,
        ]

    @torch.inference_mode()
    def infer_with_context_3d(self, imgs: torch.Tensor, targets=None) -> list[dict[str, torch.Tensor]]:
        if imgs.device != self.device:
            imgs = imgs.to(self.device)

        batch_size = imgs.shape[0]
        targets = self._require_targets(targets, batch_size)

        num_frame = int(getattr(self.detector, "num_frame", 1))
        img_inputs = self._stack_temporal_camera_inputs(targets, num_frame=num_frame)
        img_inputs = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in img_inputs]

        radar_points = [t["radar_points"].to(self.device) for t in targets]
        img_metas = [t["img_metas"] for t in targets]

        simple_test = cast(Callable[..., Any], getattr(self.detector, "simple_test"))
        det_out = simple_test(
            points=None,
            img_metas=img_metas,
            radar=radar_points,
            img=img_inputs,
            rescale=False,
        )

        outputs_3d: list[dict[str, torch.Tensor]] = []
        for frame_out in det_out:
            pts_bbox = frame_out["pts_bbox"]
            boxes_tensor = _boxes3d_to_tensor(pts_bbox["boxes_3d"], device=self.device)
            scores = pts_bbox["scores_3d"].to(self.device, dtype=torch.float32)
            labels = pts_bbox["labels_3d"].to(self.device, dtype=torch.long)

            out: dict[str, torch.Tensor] = {
                "boxes_3d": boxes_tensor,
                "scores_3d": scores,
                "labels_3d": labels,
            }
            if boxes_tensor.shape[1] >= 9:
                out["velocities"] = boxes_tensor[:, 7:9]
            outputs_3d.append(out)

        return outputs_3d

    def _project_3d_to_2d(
        self,
        boxes_3d: torch.Tensor,
        scores: torch.Tensor,
        target: dict[str, Any],
    ) -> torch.Tensor:
        if boxes_3d.numel() == 0:
            return torch.zeros((0, 5), dtype=torch.float32, device=self.device)

        img_inputs = target["img_inputs"]
        intr = img_inputs["intrins"][0].to(self.device, dtype=torch.float32)
        sensor2ego = img_inputs["sensor2egos"][0].to(self.device, dtype=torch.float32)
        ego2cam = torch.linalg.inv(sensor2ego)

        corners_ego = _boxes3d_corners(boxes_3d[:, :7])
        n = corners_ego.shape[0]
        corners_h = torch.cat([corners_ego, torch.ones((n, 8, 1), device=self.device)], dim=-1)

        corners_cam_h = torch.einsum("ij,nkj->nki", ego2cam, corners_h)
        corners_cam = corners_cam_h[..., :3]
        z = corners_cam[..., 2]

        valid = z > 1e-3
        uv_h = torch.einsum("ij,nkj->nki", intr, corners_cam)
        uv = uv_h[..., :2] / (z[..., None] + 1e-6)

        h, w, _ = target["img_metas"]["img_shape"]
        x1 = torch.where(valid, uv[..., 0], torch.full_like(uv[..., 0], 1e6)).min(dim=1).values
        y1 = torch.where(valid, uv[..., 1], torch.full_like(uv[..., 1], 1e6)).min(dim=1).values
        x2 = torch.where(valid, uv[..., 0], torch.full_like(uv[..., 0], -1e6)).max(dim=1).values
        y2 = torch.where(valid, uv[..., 1], torch.full_like(uv[..., 1], -1e6)).max(dim=1).values

        x1 = torch.clamp(x1, min=0.0, max=float(w - 1))
        y1 = torch.clamp(y1, min=0.0, max=float(h - 1))
        x2 = torch.clamp(x2, min=0.0, max=float(w - 1))
        y2 = torch.clamp(y2, min=0.0, max=float(h - 1))

        keep = (x2 > x1) & (y2 > y1)
        if not keep.any():
            return torch.zeros((0, 5), dtype=torch.float32, device=self.device)

        bboxes = torch.stack([x1[keep], y1[keep], x2[keep], y2[keep], scores[keep]], dim=-1)
        return bboxes

    @torch.inference_mode()
    def infer_with_context(self, imgs: torch.Tensor, targets=None):
        batch_size = int(imgs.shape[0])
        targets_list = self._require_targets(targets, batch_size)
        outputs_3d = self.infer_with_context_3d(imgs, targets_list)

        if not self.project_to_2d_fallback:
            raise RuntimeError(
                "RCBEVDet primary output is 3D. Use infer_with_context_3d() "
                "or enable project_to_2d_fallback for legacy 2D pipeline compatibility."
            )

        batch_bboxes: list[torch.Tensor] = []
        batch_labels: list[torch.Tensor] = []
        batch_covs: list[torch.Tensor] = []

        for out3d, target in zip(outputs_3d, targets_list):
            boxes_3d = out3d["boxes_3d"]
            scores = out3d["scores_3d"]
            labels = out3d["labels_3d"]
            bboxes2d = self._project_3d_to_2d(boxes_3d, scores, target)

            if bboxes2d.shape[0] == 0:
                batch_bboxes.append(torch.zeros((0, 5), dtype=torch.float32, device=self.device))
                batch_labels.append(torch.zeros((0,), dtype=torch.long, device=self.device))
                batch_covs.append(torch.zeros((0, 4, 4), dtype=torch.float32, device=self.device))
                continue

            n = bboxes2d.shape[0]
            covs = torch.eye(4, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(n, 1, 1)
            batch_bboxes.append(bboxes2d)
            batch_labels.append(labels[:n] if labels.shape[0] == n else labels.new_zeros((n,), dtype=torch.long))
            batch_covs.append(covs)

        return batch_bboxes, batch_labels, batch_covs

    @torch.inference_mode()
    def infer(self, _imgs: torch.Tensor):
        raise RuntimeError("RCBEVDet requires context targets. Use infer_with_context() or infer_with_context_3d().")


def factory(
    device: str = "cuda",
    checkpoint_path: Optional[str] = None,
    config_path: Optional[str] = None,
    project_to_2d_fallback: bool = True,
    smoke_eval: bool = False,
    bev_pool_backend: Optional[str] = None,
):
    if config_path is None:
        config_path = str(SRC_ROOT / "configs" / "rcbevdet" / "rcbevdet_r50_nuscenes_infer.py")

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"RCBEVDet config not found: {config_file}")

    checkpoint_file: Optional[Path] = None
    if checkpoint_path is not None:
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"RCBEVDet checkpoint not found: {checkpoint_file}")
    elif not smoke_eval:
        raise ValueError(
            "checkpoint_path is required for real RCBEVDet integration. "
            "Use smoke_eval=True only for non-metric smoke tests."
        )
    else:
        print(
            "[RCBEVDet] WARNING: smoke_eval=True and no checkpoint provided. "
            "Running with randomly initialized weights; metrics are not meaningful."
        )

    cfg = Config.fromfile(str(config_file))
    if bev_pool_backend is not None:
        if bev_pool_backend not in {"auto", "cuda_ext", "torch"}:
            raise ValueError(f"Invalid bev_pool_backend: {bev_pool_backend}")
        if "detector" in cfg.model:
            cfg.model.detector.img_view_transformer.bev_pool_backend = bev_pool_backend
        else:
            cfg.model.img_view_transformer.bev_pool_backend = bev_pool_backend
    detector_cfg = cfg.model.detector if "detector" in cfg.model else cfg.model

    detector = build_detector(detector_cfg)

    if checkpoint_file is not None:
        checkpoint = torch.load(str(checkpoint_file), map_location="cpu")
        state_dict = _unwrap_state_dict(checkpoint)
        missing, unexpected = detector.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[RCBEVDet] Warning: missing keys={len(missing)}")
        if unexpected:
            print(f"[RCBEVDet] Warning: unexpected keys={len(unexpected)}")

    classes = list(getattr(cfg, "classes", []))
    if not classes:
        classes = [
            "car",
            "truck",
            "construction_vehicle",
            "bus",
            "trailer",
            "barrier",
            "motorcycle",
            "bicycle",
            "pedestrian",
            "traffic_cone",
        ]

    return RCBEVDetWrapper(
        detector=detector,
        classes=classes,
        device=device,
        project_to_2d_fallback=project_to_2d_fallback,
    )
