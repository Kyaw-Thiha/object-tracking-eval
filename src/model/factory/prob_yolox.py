from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

from .utils import get_bboxes_from_detector, load_detector_from_checkpoint

SRC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SRC_ROOT.parent


class ProbYOLOXModelWrapper(nn.Module):

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

        results_list = get_bboxes_from_detector(self.detector, imgs)

        batch_bboxes: List[torch.Tensor] = []
        batch_labels: List[torch.Tensor] = []
        batch_covs:   List[torch.Tensor] = []

        for (det_bboxes, det_bbox_covs, det_labels, _det_score_vars) in results_list:

            batch_bboxes.append(det_bboxes)
            batch_labels.append(det_labels.to(torch.long))
            batch_covs.append(det_bbox_covs)

        return batch_bboxes, batch_labels, batch_covs

    def get_classes(self):
        return self.classes


def factory(device: str):
    config_path = SRC_ROOT / "configs" / "yolox" / "prob_yolox_x_es_mot17-half.py"
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "prob_yolox_camel" / "epoch_26.pth"

    detector = load_detector_from_checkpoint(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        device=device,
    )

    # --- COCO style dataset classes ---
    class_names = ('person', 'bicycle', 'car', 'horse')

    model = ProbYOLOXModelWrapper(detector=detector, classes=class_names, device=device)
    return model
