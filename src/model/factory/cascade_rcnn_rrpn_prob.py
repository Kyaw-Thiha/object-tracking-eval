"""Probabilistic Cascade R-CNN + RRPN wrapper with detector-native covariance support."""

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector

SRC_ROOT = Path(__file__).resolve().parents[2]


def _default_img_metas(imgs: torch.Tensor) -> list[dict[str, Any]]:
    batch_size, _, h, w = imgs.shape
    return [
        {
            "img_shape": (h, w, 3),
            "ori_shape": (h, w, 3),
            "pad_shape": (h, w, 3),
            "scale_factor": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            "flip": False,
            "flip_direction": None,
        }
        for _ in range(batch_size)
    ]


def _normalize_proposals(
    proposals: Optional[torch.Tensor | Sequence[torch.Tensor]],
    batch_size: int,
    device: torch.device,
) -> list[torch.Tensor]:
    if proposals is None:
        return [torch.zeros((0, 4), device=device, dtype=torch.float32) for _ in range(batch_size)]

    if isinstance(proposals, torch.Tensor):
        if proposals.ndim == 2 and batch_size == 1:
            return [proposals.to(device=device, dtype=torch.float32)]
        if proposals.ndim != 3:
            raise ValueError(f"Expected proposals tensor to have rank 3, got shape {tuple(proposals.shape)}")
        if proposals.shape[0] != batch_size:
            raise ValueError(f"Proposal batch {proposals.shape[0]} does not match image batch {batch_size}")
        return [proposals[i].to(device=device, dtype=torch.float32) for i in range(batch_size)]

    proposal_list: list[torch.Tensor] = []
    for item in proposals:
        if isinstance(item, torch.Tensor):
            proposal_list.append(item.to(device=device, dtype=torch.float32))
        else:
            proposal_list.append(torch.as_tensor(item, dtype=torch.float32, device=device))
    if len(proposal_list) != batch_size:
        raise ValueError(f"Proposal list length {len(proposal_list)} does not match image batch {batch_size}")
    return proposal_list


class ProbabilisticCascadeRCNNRRPN(nn.Module):
    """Two-stage detector wrapper.

    Uses detector-native covariance outputs (end-to-end path).
    """

    def __init__(self, detector: nn.Module, classes: list, device: str):
        super().__init__()
        self.detector = detector
        self.device = torch.device(device)
        self.detector.to(self.device)
        self.detector.eval()
        self.classes = classes

    def forward(
        self,
        imgs: torch.Tensor,
        proposals: Optional[torch.Tensor | Sequence[torch.Tensor]] = None,
        img_metas: Optional[Sequence[dict[str, Any]]] = None,
    ):
        return self.infer(imgs, proposals=proposals, img_metas=img_metas)

    def infer_with_context(self, imgs: torch.Tensor, targets=None):
        proposals: Optional[list[torch.Tensor]] = None
        img_metas: Optional[list[dict[str, Any]]] = None
        if isinstance(targets, list) and len(targets) > 0:
            proposals = [cast(torch.Tensor, t.get("proposals", torch.zeros((0, 4), dtype=torch.float32))) for t in targets]
            img_metas = [cast(dict[str, Any], t.get("img_metas", {})) for t in targets]
        return self.forward(imgs, proposals=proposals, img_metas=img_metas)

    @torch.inference_mode()
    def infer(
        self,
        imgs: torch.Tensor,
        proposals: Optional[torch.Tensor | Sequence[torch.Tensor]] = None,
        img_metas: Optional[Sequence[dict[str, Any]]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        if imgs.device != self.device:
            imgs = imgs.to(self.device)

        batch_size = imgs.shape[0]
        proposal_list = _normalize_proposals(proposals, batch_size=batch_size, device=self.device)
        metas = list(img_metas) if img_metas is not None else _default_img_metas(imgs)
        if len(metas) != batch_size:
            raise ValueError(f"img_metas length {len(metas)} does not match image batch {batch_size}")

        detector = cast(Any, self.detector)
        feats = detector.extract_feat(imgs)

        if not (hasattr(detector, "roi_head") and hasattr(detector.roi_head, "simple_test_with_cov")):
            raise RuntimeError(
                "Detector roi_head must implement simple_test_with_cov for end-to-end probabilistic inference."
            )

        det_bboxes, det_labels, det_covs = detector.roi_head.simple_test_with_cov(
            feats, proposal_list, metas, rescale=False
        )
        return det_bboxes, det_labels, det_covs

    def get_classes(self):
        return self.classes


def factory(device: str, checkpoint_path: Optional[Path] = None, config_path: Optional[Path] = None):
    """Factory function for probabilistic RRPN wrapper."""
    if config_path is None:
        config_path = SRC_ROOT / "configs" / "cascade_rcnn" / "cascade_rcnn_r50_rrpn_prob_nuscenes.py"

    cfg = Config.fromfile(str(config_path))
    detector_cfg = cfg.model.detector if "detector" in cfg.model else cfg.model
    detector = build_detector(detector_cfg, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))

    if checkpoint_path:
        load_checkpoint(detector, str(checkpoint_path), map_location=device)
    else:
        print("[ProbCascadeRCNN] No checkpoint provided, using random initialization")

    classes = (
        cfg.classes
        if hasattr(cfg, "classes")
        else ["car", "truck", "bus", "person", "bicycle", "motorcycle", "construction_vehicle", "trailer", "movable_object", "traffic_cone"]
    )
    return ProbabilisticCascadeRCNNRRPN(detector, classes, device)
