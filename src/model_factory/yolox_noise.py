import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import math

from mmcv import Config
from mmdet.models import build_detector

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent

from scipy.optimize import linear_sum_assignment


class ProbYOLOXModelWrapper(nn.Module):

    def __init__(self, detector: nn.Module, classes: list, device: str):
        super().__init__()
        self.detector = detector
        self.device = torch.device(device)
        self.detector.to(self.device)
        self.detector.eval()
        self.classes = classes
        self.n_repeat = 5  # number of noise samples per image
        self.noise_ratio = 0.1  # noise ratio

    @torch.inference_mode()
    def infer(self, imgs: torch.Tensor):

        if imgs.device != self.device:
            imgs = imgs.to(self.device)

        B, _, H, W = imgs.shape

        batch_bboxes = []
        batch_labels = []
        batch_covs = []

        n_repeat_bboxes = [[] for _ in range(B)]
        n_repeat_labels = [[] for _ in range(B)]

        # print("[INFER DEBUG] Start Inference on batch size of ", B)

        for r in range(self.n_repeat):
            
            # print(f"[INFER DEBUG] repeat #{r}")
            noise = torch.randn_like(imgs) * (self.noise_ratio * imgs.std())
            noisy_imgs = imgs + noise

            feats = self.detector.extract_feat(noisy_imgs)
            cls_scores, cls_vars, bbox_preds, bbox_covs, objectnesses = self.detector.bbox_head(feats)

            img_metas = [
                dict(
                    img_shape=(H, W, 3),
                    ori_shape=(H, W, 3),
                    pad_shape=(H, W, 3),
                    scale_factor=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                    flip=False,
                    flip_direction=None,
                )
                for _ in range(B)
            ]

            results_list = self.detector.bbox_head.get_bboxes(
                cls_scores=cls_scores,
                cls_vars=cls_vars,
                bbox_preds=bbox_preds,
                bbox_covs=bbox_covs,
                objectnesses=objectnesses,
                img_metas=img_metas,
                cfg=self.detector.bbox_head.test_cfg,
                rescale=False,
                with_nms=True,
            )

            # print(f"[INFER DEBUG] got result_list")

            for i, (det_bboxes, det_bbox_covs, det_labels, _det_score_vars) in enumerate(results_list):
                n_repeat_bboxes[i].append(det_bboxes)
                n_repeat_labels[i].append(det_labels)
            
        for i in range(B):
            bboxes, labels = n_repeat_bboxes[i], n_repeat_labels[i]
            # print(f"[INFER DEBUG] start compute mean covs for img #{i} in batch")
            mean_bboxes, majority_labels, covariances = self.compute_mean_covariance(bboxes, labels)

            # print("mean_bboxes:", mean_bboxes)
            # prepare tensors for tracker
            det_bboxes = torch.tensor(mean_bboxes, dtype=torch.float32, device=self.device)
            # print("det_bboxes tensor shape:", det_bboxes.shape)
            det_labels = torch.tensor(majority_labels, dtype=torch.long, device=self.device)
            # print("det_labels tensor shape:", det_labels.shape)
            bbox_covs = torch.tensor(covariances, dtype=torch.float32, device=self.device)
            # print("bbox_covs tensor shape:", bbox_covs.shape)

            # print("correct cov matrices shape: ", torch.eye(4, device=self.device).unsqueeze(0).repeat(det_bboxes.shape[0], 1, 1).shape)

            # --- ChatGPT fix: numpy.linalg.LinAlgError: 1-th leading minor of the array is not positive definite "ensure covariance matrices are SPD before returning" ---
            # error originates inside scipy.linalg.decomp_cholesky when covariances are not SPD
            sanitized_covs = []
            for cov in bbox_covs:
                # symmetrize to avoid tiny asymmetries
                cov = 0.5 * (cov + cov.T)
                # compute eigenvalues
                eigvals, eigvecs = torch.linalg.eigh(cov)
                # clip tiny / negative eigenvalues
                eps = 1e-3  # enough to guarantee Cholesky works
                eigvals = torch.clamp(eigvals, min=eps)
                # reconstruct SPD matrix
                cov_spd = (eigvecs @ torch.diag(eigvals) @ eigvecs.T)
                sanitized_covs.append(cov_spd)

            bbox_covs = torch.stack(sanitized_covs)

            batch_bboxes.append(det_bboxes)
            batch_labels.append(det_labels)
            batch_covs.append(bbox_covs)
        
        return batch_bboxes, batch_labels, batch_covs

    def get_classes(self):
        return self.classes
    
    def sanitize_covs(self, covs: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """
        Ensure covariance matrices are Symmetric Positive Definite (SPD).

        Supports:
        - single matrix (4,4)
        - batch (N,4,4)

        All operations stay on the same device (GPU-friendly).
        """

        # --- Case 1: Single matrix ---
        if covs.ndim == 2:
            cov = 0.5 * (covs + covs.T)
            eigvals, eigvecs = torch.linalg.eigh(cov)
            eigvals = torch.clamp(eigvals, min=eps)
            return eigvecs @ torch.diag(eigvals) @ eigvecs.T

        # --- Case 2: Batch (N,4,4) ---
        covs = 0.5 * (covs + covs.transpose(1,2))
        eigvals, eigvecs = torch.linalg.eigh(covs)    # batch Eigh

        eigvals = torch.clamp(eigvals, min=eps)

        # Reconstruct SPD: (eigvecs @ diag(eigvals) @ eigvecs^T)
        covs_spd = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(1,2)
        return covs_spd
    
    def compute_mean_covariance(self, bboxes, labels, iou_threshold=0.1):
        """
        Args:
            bboxes: list over repeats; each element may be:
                - list of Tensors [tensor(5,), tensor(5,), ...]
                - Tensor of shape (N,5)
                - Tensor of shape (5,)
                Each detection is [x, y, w, h, score] in xywh format.
            labels: list over repeats; list of lists of class IDs

        Returns:
            final_bboxes_mean: list of (x, y, w, h, score)
            final_labels:      list of majority cls_ids
            final_covariances: list of numpy (4,4) covariance matrices (over x,y,w,h)
        """

        device = self.device

        # ---------- Detection normalization helper ----------
        def normalize_dets(det_list):
            """
            Convert detection output into a list of (5,) tensors.
            Handles all YOLOX output formats.
            """
            if isinstance(det_list, torch.Tensor):
                det_list = det_list.detach()
                if det_list.ndim == 1:       # (5,)
                    return [det_list]
                elif det_list.ndim == 2:     # (N,5)
                    return [d for d in det_list]
                else:
                    raise ValueError(f"Unexpected det tensor shape: {det_list.shape}")
            else:
                # Already a list; ensure all are tensors and detached
                out = []
                for d in det_list:
                    if isinstance(d, torch.Tensor):
                        out.append(d.detach())
                    else:
                        out.append(torch.tensor(d, dtype=torch.float32))
                return out

        # ---------- Extract base repeat ----------
        base_dets = normalize_dets(bboxes[0])
        base_labels = labels[0]

        if len(base_dets) == 0:
            return [], [], []

        base_boxes = torch.stack(base_dets).to(device)   # (K,5), xywh
        K = base_boxes.shape[0]

        # Grouping structures
        grouped_boxes = [[] for _ in range(K)]
        grouped_labels = [[] for _ in range(K)]

        for i in range(K):
            grouped_boxes[i].append(base_boxes[i])
            grouped_labels[i].append(int(base_labels[i]))

        # ---------- IoU for xywh ----------
        def iou_xywh(boxA, boxB):
            """
            boxA, boxB: (..., 4) in xywh
            """
            Ax, Ay, Aw, Ah = boxA[..., 0], boxA[..., 1], boxA[..., 2], boxA[..., 3]
            Bx, By, Bw, Bh = boxB[..., 0], boxB[..., 1], boxB[..., 2], boxB[..., 3]

            Ax1, Ay1 = Ax, Ay
            Ax2, Ay2 = Ax + Aw, Ay + Ah
            Bx1, By1 = Bx, By
            Bx2, By2 = Bx + Bw, By + Bh

            xA = torch.max(Ax1, Bx1)
            yA = torch.max(Ay1, By1)
            xB = torch.min(Ax2, Bx2)
            yB = torch.min(Ay2, By2)

            inter_w = torch.clamp(xB - xA, min=0)
            inter_h = torch.clamp(yB - yA, min=0)
            inter = inter_w * inter_h

            areaA = Aw * Ah
            areaB = Bw * Bh
            union = areaA + areaB - inter + 1e-6

            return inter / union

        # ---------- MATCH REPEATS (SciPy Hungarian) ----------
        for r in range(1, len(bboxes)):
            curr_raw = bboxes[r]
            curr_labels = labels[r]

            curr = normalize_dets(curr_raw)
            if len(curr) == 0:
                continue

            curr_boxes = torch.stack(curr).to(device)  # (M,5) xywh
            M = curr_boxes.shape[0]

            # IoU in xywh → (K, M)
            ious = iou_xywh(
                base_boxes[:, :4].unsqueeze(1),   # (K,1,4)
                curr_boxes[:, :4].unsqueeze(0),   # (1,M,4)
            )  # (K,M)

            # Cost matrix for Hungarian: 1 - IoU
            cost_np = (1.0 - ious).cpu().numpy()
            # Debug if needed:
            # print("[DEBUG] Hungarian cost shape:", cost_np.shape)

            row_ind, col_ind = linear_sum_assignment(cost_np)

            for i_idx, j_idx in zip(row_ind, col_ind):
                i_idx = int(i_idx)
                j_idx = int(j_idx)
                if i_idx < K and j_idx < M and ious[i_idx, j_idx] >= iou_threshold:
                    grouped_boxes[i_idx].append(curr_boxes[j_idx])
                    grouped_labels[i_idx].append(int(curr_labels[j_idx]))

        # ---------- COMPUTE MEAN + COVARIANCE ----------
        final_bboxes_mean = []
        final_labels = []
        final_covariances = []

        for i in range(K):
            samples = grouped_boxes[i]
            labels_i = grouped_labels[i]

            if len(samples) == 0:
                continue

            samples_t = torch.stack(samples)        # (n,5)
            xywh = samples_t[:, :4]                 # (n,4)
            scores = samples_t[:, 4]

            mean_xywh = xywh.mean(dim=0)
            mean_score = scores.mean()

            mean_box = (
                float(mean_xywh[0]), float(mean_xywh[1]),
                float(mean_xywh[2]), float(mean_xywh[3]),
                float(mean_score),
            )

            most_common_label = max(set(labels_i), key=labels_i.count)

            if len(samples) > 1:
                cov = torch.cov(xywh.T)            # covariance over x,y,w,h
                cov = self.sanitize_covs(cov)      # SPD correction
                cov_np = cov.cpu().numpy()
            else:
                cov_np = np.zeros((4, 4), dtype=np.float32)

            final_bboxes_mean.append(mean_box)
            final_labels.append(most_common_label)
            final_covariances.append(cov_np)

        return final_bboxes_mean, final_labels, final_covariances


def _load_detector_from_checkpoint(config_path: str, checkpoint_path: str, device: str):
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
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "prob_yolox_camel" / "epoch_26.pth"

    detector = _load_detector_from_checkpoint(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        device=device,
    )

    # --- COCO style dataset classes ---
    class_names = ('person', 'bicycle', 'car', 'horse')

    model = ProbYOLOXModelWrapper(detector=detector, classes=class_names, device=device)
    return model
