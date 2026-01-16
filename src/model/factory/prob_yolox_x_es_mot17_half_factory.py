from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

from .utils import get_bboxes_from_detector, load_detector_from_checkpoint

SRC_ROOT = Path(__file__).resolve().parents[2]
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

        results_list = get_bboxes_from_detector(self.detector, imgs)

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
    # config_path = os.path.join(PROJECT_ROOT, "src", "configs", "yolox", "prob_yolox_x_es_mot17-half.py")
    # checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "prob_yolox_mot17", "epoch_69.pth")

    config_path = SRC_ROOT / "configs" / "yolox" / "prob_yolox_x_es_mot17-half.py"
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "prob_yolox_mot17" / "epoch_69.pth"


    detector = load_detector_from_checkpoint(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        device=device,
    )

    wrapped = ProbYOLOXModelWrapper(detector=detector, device=device)
    return wrapped
