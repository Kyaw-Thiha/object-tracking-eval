"""Probabilistic Cascade RoI Head - handles 3-output bbox heads with covariance."""

from typing import List, Tuple

import torch
from torch import Tensor
from mmdet.core import bbox2roi, multiclass_nms
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads import CascadeRoIHead


@HEADS.register_module()
class ProbabilisticCascadeRoIHead(CascadeRoIHead):
    """
    Cascade RoI head that handles probabilistic bbox heads (3 outputs).

    Works with both standard (2-output) and probabilistic (3-output) bbox heads.
    Probabilistic heads return (cls_score, bbox_pred, bbox_cov).
    Covariance is refined through all cascade stages.
    """

    def _bbox_forward(self, stage: int, x: Tuple[Tensor], rois: Tensor) -> dict:
        """
        Forward bbox head at given stage.

        Handles both 2-output (standard) and 3-output (probabilistic) heads.

        Args:
            stage: Cascade stage index (0, 1, 2)
            x: Multi-level feature maps
            rois: (N, 5) [batch_idx, x1, y1, x2, y2]

        Returns:
            bbox_results: dict with cls_score, bbox_pred, (bbox_cov), bbox_feats
        """
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]

        # Extract RoI features
        bbox_feats = bbox_roi_extractor(
            x[: bbox_roi_extractor.num_inputs], rois
        )

        # Check if probabilistic head (returns 3 outputs)
        is_prob = hasattr(bbox_head, 'with_cov') and bbox_head.with_cov

        if is_prob:
            # Probabilistic head: 3 outputs
            cls_score, bbox_pred, bbox_cov = bbox_head(bbox_feats)
            bbox_results = dict(
                cls_score=cls_score,
                bbox_pred=bbox_pred,
                bbox_cov=bbox_cov,  # Include covariance
                bbox_feats=bbox_feats,
            )
        else:
            # Standard head: 2 outputs
            cls_score, bbox_pred = bbox_head(bbox_feats)
            bbox_results = dict(
                cls_score=cls_score,
                bbox_pred=bbox_pred,
                bbox_feats=bbox_feats,
            )

        return bbox_results

    def _bbox_forward_train(
        self,
        stage: int,
        x: Tuple[Tensor],
        sampling_results: List,
        gt_bboxes: List[Tensor],
        gt_labels: List[Tensor],
        rcnn_train_cfg: dict
    ) -> dict:
        """
        Forward + loss computation for training at given stage.

        Passes covariance to loss function if probabilistic head.

        Args:
            stage: Cascade stage index
            x: Multi-level features
            sampling_results: Sampled RoIs per image
            gt_bboxes: Ground truth boxes per image
            gt_labels: Ground truth labels per image
            rcnn_train_cfg: Training config for RCNN

        Returns:
            bbox_results: dict with loss_bbox and other losses
        """
        # Convert sampling results to rois
        rois = bbox2roi([res.bboxes for res in sampling_results])

        # Forward through bbox head
        bbox_results = self._bbox_forward(stage, x, rois)

        # Get targets for loss computation
        bbox_targets = self.bbox_head[stage].get_targets(
            sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg
        )

        # Compute losses
        bbox_head = self.bbox_head[stage]
        is_prob = hasattr(bbox_head, 'with_cov') and bbox_head.with_cov

        if is_prob:
            # Pass covariance to loss (YOLOX pattern)
            loss_bbox = bbox_head.loss(
                bbox_results['cls_score'],
                bbox_results['bbox_pred'],
                bbox_results['bbox_cov'],  # Pass covariance!
                rois,
                *bbox_targets
            )
        else:
            # Standard loss (no covariance)
            loss_bbox = bbox_head.loss(
                bbox_results['cls_score'],
                bbox_results['bbox_pred'],
                rois,
                *bbox_targets
            )

        bbox_results.update(
            loss_bbox=loss_bbox,
            rois=rois,
            bbox_targets=bbox_targets,
        )
        return bbox_results

    @torch.no_grad()
    def simple_test_with_cov(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation and return detection covariances."""
        assert self.with_bbox, "Bbox head must be implemented."
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta["img_shape"] for meta in img_metas)
        scale_factors = tuple(meta["scale_factor"] for meta in img_metas)
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)
        if rois.shape[0] == 0:
            empty_bbox = torch.zeros((0, 5), dtype=torch.float32, device=x[0].device)
            empty_label = torch.zeros((0,), dtype=torch.long, device=x[0].device)
            empty_cov = torch.zeros((0, 4, 4), dtype=torch.float32, device=x[0].device)
            return [empty_bbox.clone() for _ in range(num_imgs)], [empty_label.clone() for _ in range(num_imgs)], [empty_cov.clone() for _ in range(num_imgs)]

        ms_scores = []
        bbox_cov = None
        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, x, rois)
            cls_score = bbox_results["cls_score"]
            bbox_pred = bbox_results["bbox_pred"]
            bbox_cov = bbox_results.get("bbox_cov", None)

            num_proposals_per_img = tuple(len(proposals) for proposals in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head[i].bbox_pred_split(bbox_pred, num_proposals_per_img)
            if isinstance(bbox_cov, torch.Tensor):
                bbox_cov = bbox_cov.split(num_proposals_per_img, 0)
            else:
                bbox_cov = [None for _ in range(num_imgs)]
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                if self.bbox_head[i].custom_activation:
                    cls_score = [self.bbox_head[i].loss_cls.get_activation(s) for s in cls_score]
                refine_rois_list = []
                for j in range(num_imgs):
                    if rois[j].shape[0] > 0:
                        bbox_label = cls_score[j][:, :-1].argmax(dim=1)
                        refined_rois = self.bbox_head[i].regress_by_class(rois[j], bbox_label, bbox_pred[j], img_metas[j])
                        refine_rois_list.append(refined_rois)
                rois = torch.cat(refine_rois_list)

        cls_score = [sum(score[i] for score in ms_scores) / float(len(ms_scores)) for i in range(num_imgs)]

        det_bboxes = []
        det_labels = []
        det_covs = []
        final_bbox_head = self.bbox_head[-1]
        num_classes = final_bbox_head.num_classes

        for i in range(num_imgs):
            decoded_bboxes, scores = final_bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=None,
            )
            det_bbox, det_label, flat_inds = multiclass_nms(
                decoded_bboxes,
                scores,
                rcnn_test_cfg.score_thr,
                rcnn_test_cfg.nms,
                rcnn_test_cfg.max_per_img,
                return_inds=True,
            )

            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

            if bbox_cov is None or bbox_cov[i] is None or det_bbox.shape[0] == 0:
                det_covs.append(torch.zeros((det_bbox.shape[0], 4, 4), dtype=torch.float32, device=det_bbox.device))
                continue

            roi_inds = torch.div(flat_inds, num_classes, rounding_mode="floor")
            img_bbox_cov = bbox_cov[i]
            if final_bbox_head.reg_class_agnostic:
                selected_log_vars = img_bbox_cov.view(-1, 4)[roi_inds]
            else:
                selected_log_vars = img_bbox_cov.view(-1, num_classes, 4)[roi_inds, det_label]

            selected_vars = torch.exp(selected_log_vars.clamp(min=-10.0, max=10.0))
            det_covs.append(torch.diag_embed(selected_vars))

        return det_bboxes, det_labels, det_covs
