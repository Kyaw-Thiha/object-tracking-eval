import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from mmcv import Config
from mmdet.models import build_detector

SRC_ROOT = Path(__file__).resolve().parents[1]


class IdentityCovYOLOXModelWrapper(nn.Module):

    def __init__(self, detector: nn.Module, classes: list, device: str):
        super().__init__()
        self.detector = detector
        self.device = torch.device(device)
        self.detector.to(self.device)
        self.detector.eval()
        self.classes = classes

    @torch.inference_mode()
    def infer(
        self, imgs: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            imgs: images tensor, (B, 3, H, W) torch.Tensor

        Returns:
            (batch_bboxes, batch_labels, batch_covs)
                batch_bboxes: list[Tensor], each (N_i, 5)
                batch_labels: list[Tensor], each (N_i,)
                batch_covs:   list[Tensor], each (N_i, 4, 4)
        """
        if imgs.device != self.device:
            imgs = imgs.to(self.device)

        B, _, H, W = imgs.shape

        feats = self.detector.extract_feat(imgs)

        cls_scores, cls_vars, bbox_preds, bbox_covs, objectnesses = \
            self.detector.bbox_head(feats)

        img_metas = []
        for _ in range(B):
            meta = dict(
                img_shape=(H, W, 3),
                ori_shape=(H, W, 3),
                pad_shape=(H, W, 3),
                scale_factor=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                flip=False,
                flip_direction=None,
            )
            img_metas.append(meta)

        results_list = self.detector.bbox_head.get_bboxes(
            cls_scores=cls_scores,
            cls_vars=cls_vars,
            bbox_preds=bbox_preds,
            bbox_covs=bbox_covs,
            objectnesses=objectnesses,
            img_metas=img_metas,
            cfg=self.detector.bbox_head.test_cfg
            if hasattr(self.detector.bbox_head, "test_cfg")
            else getattr(self.detector, "test_cfg", None),
            rescale=False,
            with_nms=True,
        )

        batch_bboxes: List[torch.Tensor] = []
        batch_labels: List[torch.Tensor] = []
        batch_covs:   List[torch.Tensor] = []

        # Fixed Identity Covariance matrix
        # eye_covs = torch.eye(4, device=imgs.device).unsqueeze(0)

        

        for (det_bboxes, det_bbox_covs, det_labels, _det_score_vars) in results_list:

            batch_bboxes.append(det_bboxes)
            batch_labels.append(det_labels.to(torch.long))
            
            # create identity covariance matrices
            identity_covs = torch.eye(4, device=imgs.device).unsqueeze(0).repeat(det_bbox_covs.shape[0], 1, 1)

            batch_covs.append(identity_covs)

        return batch_bboxes, batch_labels, batch_covs

    def get_classes(self):
        return self.classes


def _load_detector_from_checkpoint(
    config_path: str,
    checkpoint_path: str,
    device: str,
) -> nn.Module:
    # build detector from config file
    cfg = Config.fromfile(config_path)
    detector_cfg = cfg.model.detector
    detector = build_detector(detector_cfg)

    # load checkpoint
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("detector."):
            new_k = k[len("detector.") :]
        elif k.startswith("model."):
            new_k = k[len("model.") :]
        else:
            new_k = k
        new_state_dict[new_k] = v

    missing, unexpected = detector.load_state_dict(new_state_dict, strict=False)
    if missing:
        print("[ProbYOLOX factory] Warning: missing keys when loading state_dict:")
        for k in missing:
            print("   ", k)
    if unexpected:
        print("[ProbYOLOX factory] Warning: unexpected keys when loading state_dict:")
        for k in unexpected:
            print("   ", k)

    detector.to(torch.device(device))
    detector.eval()
    return detector


def factory(device: str):
    config_path = SRC_ROOT / "configs" / "yolox" / "prob_yolox_x_es_mot17-half.py"
    checkpoint_path = SRC_ROOT / "checkpoints" / "prob_yolox_camel" / "epoch_26.pth"

    detector = _load_detector_from_checkpoint(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        device=device,
    )

    # --- COCO style dataset classes ---
    class_names = ('person', 'bicycle', 'car', 'horse')

    model = IdentityCovYOLOXModelWrapper(detector=detector, classes=class_names, device=device)

    return model
