"""Probabilistic bbox head for Cascade R-CNN - predicts covariance (YOLOX-style)."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.bbox_heads import Shared2FCBBoxHead


@HEADS.register_module()
class ProbabilisticBBoxHead(Shared2FCBBoxHead):
    """
    Bbox head with covariance prediction following YOLOX pattern.

    Returns 3 outputs: (cls_score, bbox_pred, bbox_cov)
    Uses NLL loss for bbox regression with uncertainty.

    Args:
        with_cov: Whether to predict covariance (default: True)
        loss_bbox: Loss config (should be NLL for probabilistic training)
    """

    def __init__(
        self,
        *args,
        with_cov: bool = True,
        loss_bbox: dict = dict(type='NLL', covariance_type='diagonal', loss_type='L1'),
        **kwargs
    ):
        self.with_cov = with_cov
        super().__init__(*args, loss_bbox=loss_bbox, **kwargs)

        # ConvFCBBoxHead/Shared2FCBBoxHead build layers in __init__ and do not call
        # a subclass _init_layers(). Define the covariance head here so it always exists.
        if self.with_cov and self.with_reg:
            cov_dim = 4 if self.reg_class_agnostic else 4 * self.num_classes
            self.fc_cov = nn.Linear(self.reg_last_dim, cov_dim)
            nn.init.constant_(self.fc_cov.weight, 0)
            nn.init.constant_(self.fc_cov.bias, 0)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Forward pass returning 3 outputs (YOLOX pattern).

        Args:
            x: Input features (N, C, H, W) from RoI extractor

        Returns:
            cls_score: (N, num_classes)
            bbox_pred: (N, 4) or (N, 4*num_classes)
            bbox_cov: (N, 4) log-variances (None if with_cov=False)
        """
        # Flatten RoI features if needed (N, C, H, W) -> (N, C*H*W)
        if x.dim() > 2:
            x = x.flatten(1)

        # Shared layers
        if self.num_shared_fcs > 0:
            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        x_cls = x
        x_reg = x

        # Classification branch
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))
        cls_score = self.fc_cls(x_cls) if self.with_cls else None

        # Regression branch
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        # Covariance branch (parallel to regression)
        bbox_cov = None
        if self.with_cov and self.with_reg:
            if not hasattr(self, "fc_cov"):
                raise RuntimeError("ProbabilisticBBoxHead expected fc_cov but it is not initialized")
            bbox_cov = self.fc_cov(x_reg)  # (N, 4) log-variances

        return cls_score, bbox_pred, bbox_cov

    def loss(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        bbox_cov: Optional[Tensor],
        rois: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        bbox_targets: Tensor,
        bbox_weights: Tensor,
        reduction_override: Optional[str] = None,
    ) -> dict:
        """
        Loss computation with NLL for probabilistic bbox regression.

        Args:
            cls_score: (N, num_classes)
            bbox_pred: (N, 4) or (N, 4*num_classes)
            bbox_cov: (N, 4) log-variances (can be None)
            rois, labels, label_weights, bbox_targets, bbox_weights: standard targets

        Returns:
            losses: dict with 'loss_cls' and 'loss_bbox'
        """
        losses = dict()

        # Classification loss (standard)
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.0)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override
                )
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_

        # Bbox regression loss with covariance (NLL)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)

            # Do not perform regression on background
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                    # is applied directly on the decoded bounding boxes, both
                    # the predicted boxes and the target boxes should be with
                    # absolute coordinate format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)

                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds]
                else:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)[
                        pos_inds, labels[pos_inds]
                    ]

                # NLL loss with covariance
                if bbox_cov is not None and self.with_cov:
                    # Extract covariance for positive samples
                    if self.reg_class_agnostic:
                        pos_bbox_cov = bbox_cov.view(bbox_cov.size(0), 4)[pos_inds]
                    else:
                        pos_bbox_cov = bbox_cov.view(bbox_cov.size(0), -1, 4)[
                            pos_inds, labels[pos_inds]
                        ]

                    # Call NLL loss: loss(pred, pred_cov, target, weight=...)
                    losses['loss_bbox'] = self.loss_bbox(
                        pos_bbox_pred,
                        pos_bbox_cov,  # Pass covariance (YOLOX pattern)
                        bbox_targets[pos_inds],
                        weight=bbox_weights[pos_inds],  # Use keyword argument
                        avg_factor=bbox_targets.size(0),
                        reduction_override=reduction_override
                    )
                else:
                    # Fallback to standard loss (no covariance)
                    losses['loss_bbox'] = self.loss_bbox(
                        pos_bbox_pred,
                        bbox_targets[pos_inds],
                        bbox_weights[pos_inds],
                        avg_factor=bbox_targets.size(0),
                        reduction_override=reduction_override
                    )
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()

        return losses
