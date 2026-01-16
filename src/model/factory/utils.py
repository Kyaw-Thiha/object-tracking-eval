import os
from typing import List

import numpy as np
import torch
from mmcv import Config
from mmdet.models import build_detector


def build_img_metas(batch_size: int, height: int, width: int) -> List[dict]:
    """Build minimal MMDet-style img_metas for inference without rescaling."""
    img_metas = []
    for _ in range(batch_size):
        # We keep rescale=False here, so scale_factor is not used.
        img_metas.append(
            dict(
                img_shape=(height, width, 3),
                ori_shape=(height, width, 3),
                pad_shape=(height, width, 3),
                scale_factor=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                flip=False,
                flip_direction=None,
            )
        )
    return img_metas


def get_bboxes_from_detector(detector, imgs: torch.Tensor):
    """Run the detector head on a batch and return post-NMS bboxes/labels/covs."""
    batch_size, _, height, width = imgs.shape

    # ---- 1. Forward backbone + neck ----
    feats = detector.extract_feat(imgs)

    # ---- 2. Forward head to get raw predictions ----
    # ProbabilisticYOLOXHead2.forward returns:
    #   cls_scores, cls_vars, bbox_preds, bbox_covs, objectnesses
    cls_scores, cls_vars, bbox_preds, bbox_covs, objectnesses = detector.bbox_head(feats)

    # ---- 3. Build dummy img_metas for each image ----
    img_metas = build_img_metas(batch_size, height, width)

    # ---- 4. Use the head's probabilistic get_bboxes() ----
    # This is where mean + covariance are computed and NMS / fusion is done.
    # get_bboxes() returns, per image:
    #   (det_bboxes, det_bbox_covs, det_labels, det_score_vars)
    cfg = detector.bbox_head.test_cfg if hasattr(detector.bbox_head, "test_cfg") else getattr(detector, "test_cfg", None)
    return detector.bbox_head.get_bboxes(
        cls_scores=cls_scores,
        cls_vars=cls_vars,
        bbox_preds=bbox_preds,
        bbox_covs=bbox_covs,
        objectnesses=objectnesses,
        img_metas=img_metas,
        cfg=cfg,
        rescale=False,
        with_nms=True,
    )


def load_detector_from_checkpoint(
    config_path: str,
    checkpoint_path: str,
    device: str,
):
    """
    Build the underlying detector from the mmdet-style config + your trained .pth.
    """
    # In your config, the actual detector is under model.detector
    cfg = Config.fromfile(config_path)
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
