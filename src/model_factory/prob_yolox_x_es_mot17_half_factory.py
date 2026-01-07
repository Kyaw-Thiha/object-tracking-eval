import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from mmcv import Config
from mmdet.models import build_detector

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent


class ProbYOLOXModelWrapper(nn.Module):
    """
    Wrap the trained Probabilistic YOLOX detector so that it exposes:

        infer(imgs) -> (batch_bboxes, batch_labels, batch_covs)

    where:
        - imgs: torch.Tensor of shape (B, 3, H, W) on the correct device
        - batch_bboxes: list[Tensor], each (Ni, 5) = [x1, y1, x2, y2, score]
        - batch_labels: list[Tensor], each (Ni,) = class indices (long)
        - batch_covs:   list[Tensor], each (Ni, 4, 4) = bbox covariance
    """

    def __init__(self, detector: nn.Module, device: str):
        super().__init__()
        self.detector = detector
        self.device = torch.device(device)
        self.detector.to(self.device)
        self.detector.eval()

        # sanity: make sure we have the pieces we need
        assert hasattr(self.detector, "extract_feat"), \
            "Detector must have extract_feat(imgs) -> feature maps"
        assert hasattr(self.detector, "bbox_head"), \
            "Detector must have bbox_head"
        assert hasattr(self.detector.bbox_head, "get_bboxes"), \
            "bbox_head must implement get_bboxes (ProbabilisticYOLOXHead2)"

    @torch.inference_mode()
    def infer(
        self, imgs: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            imgs: (B, 3, H, W) torch.Tensor on either CPU or GPU

        Returns:
            (batch_bboxes, batch_labels, batch_covs)
                batch_bboxes: list[Tensor], each (N_i, 5)
                batch_labels: list[Tensor], each (N_i,)
                batch_covs:   list[Tensor], each (N_i, 4, 4)
        """
        # Ensure imgs are on the same device as the detector
        if imgs.device != self.device:
            imgs = imgs.to(self.device)

        B, _, H, W = imgs.shape

        # ---- 1. Forward backbone + neck ----
        feats = self.detector.extract_feat(imgs)

        # ---- 2. Forward head to get raw predictions ----
        # ProbabilisticYOLOXHead2.forward returns:
        #   cls_scores, cls_vars, bbox_preds, bbox_covs, objectnesses
        cls_scores, cls_vars, bbox_preds, bbox_covs, objectnesses = \
            self.detector.bbox_head(feats)

        # ---- 3. Build dummy img_metas for each image ----
        # We keep rescale=False here, so scale_factor is not used.
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

        # ---- 4. Use the head's probabilistic get_bboxes() ----
        # This is where mean + covariance are computed and NMS / fusion is done.
        # get_bboxes() returns, per image:
        #   (det_bboxes, det_bbox_covs, det_labels, det_score_vars)
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

        # ---- 5. Convert to the format your evaluation pipeline expects ----
        batch_bboxes: List[torch.Tensor] = []
        batch_labels: List[torch.Tensor] = []
        batch_covs:   List[torch.Tensor] = []

        for (det_bboxes, det_bbox_covs, det_labels, _det_score_vars) in results_list:
            # det_bboxes: (N, 5) [x1, y1, x2, y2, score]
            # det_bbox_covs: (N, 4, 4)
            # det_labels: (N,)
            # All already torch.Tensors on the correct device.
            batch_bboxes.append(det_bboxes)
            batch_labels.append(det_labels.to(torch.long))
            batch_covs.append(det_bbox_covs)

        return batch_bboxes, batch_labels, batch_covs

    def get_classes(self):
        # For your MOT17 setup, only 'pedestrian'
        # (this must match what you trained with)
        return ("pedestrian",)


def _load_detector_from_checkpoint(
    config_path: str,
    checkpoint_path: str,
    device: str,
) -> nn.Module:
    """
    Build the underlying detector from the mmdet-style config + your trained .pth.
    """
    cfg = Config.fromfile(config_path)

    # In your config, the actual detector is under model.detector
    detector_cfg = cfg.model.detector

    # Build detector (SingleStageDetector / YOLOX-like)
    detector = build_detector(detector_cfg)

    # ---- Load weights ----
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    # Some trainings save weights under prefixes like "detector." or "model."
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
    """
    Entry point used by your evaluation_pipeline.py:

        factory = import_model_factory(args.model_factory)
        model = factory(device=device)

    Returns:
        ProbYOLOXModelWrapper instance with .infer() and .get_classes()
    """
    # Paths based on your training command:
    #   python train.py configs/yolox/prob_yolox_x_es_mot17-half.py
    # config_path = os.path.join(
    #     "/home/allynbao/project/UncertaintyTrack/src",
    #     "configs/yolox/prob_yolox_x_es_mot17-half.py",
    # )
    # checkpoint_path = os.path.join(
    #     "/home/allynbao/project/UncertaintyTrack/src",
    #     "work_dirs/prob_yolox_x_es_mot17-half/epoch_69.pth",
    # )

    config_path = SRC_ROOT / "configs" / "yolox" / "prob_yolox_x_es_mot17-half.py"
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "prob_yolox_mot17" / "epoch_69.pth"


    detector = _load_detector_from_checkpoint(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        device=device,
    )

    wrapped = ProbYOLOXModelWrapper(detector=detector, device=device)
    return wrapped
