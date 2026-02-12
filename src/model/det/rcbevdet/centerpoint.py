"""Local CenterPoint-like detector base for RCBEVDet (no mmdet3d)."""

from __future__ import annotations

from typing import Any, Sequence, cast

import torch
from torch import nn
from mmcv.ops import Voxelization

from . import builder


def bbox3d2result(bboxes, scores, labels, attrs=None):
    result_dict = {
        "boxes_3d": bboxes.to("cpu"),
        "scores_3d": scores.cpu(),
        "labels_3d": labels.cpu(),
    }
    if attrs is not None:
        result_dict["attrs_3d"] = attrs.cpu()
    return result_dict


class CenterPoint(nn.Module):
    """Subset of CenterPoint behavior used by migrated RCBEVDet classes."""

    def __init__(
        self,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        pts_neck=None,
        pts_bbox_head=None,
        pts_seg_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del kwargs
        self.pts_voxel_layer = (
            Voxelization(**pts_voxel_layer) if isinstance(pts_voxel_layer, dict) else pts_voxel_layer
        )
        self.pts_voxel_encoder = (
            builder.build_voxel_encoder(pts_voxel_encoder)
            if isinstance(pts_voxel_encoder, dict) else pts_voxel_encoder
        )
        self.pts_middle_encoder = (
            builder.build_middle_encoder(pts_middle_encoder)
            if isinstance(pts_middle_encoder, dict) else pts_middle_encoder
        )
        self.pts_fusion_layer = pts_fusion_layer
        self.img_backbone = (
            builder.build_backbone(img_backbone)
            if isinstance(img_backbone, dict) else img_backbone
        )
        self.pts_backbone = (
            builder.build_backbone(pts_backbone)
            if isinstance(pts_backbone, dict) else pts_backbone
        )
        self.img_neck = (
            builder.build_neck(img_neck)
            if isinstance(img_neck, dict) else img_neck
        )
        self.pts_neck = (
            builder.build_neck(pts_neck)
            if isinstance(pts_neck, dict) else pts_neck
        )
        self.pts_bbox_head = (
            builder.build_head(pts_bbox_head)
            if isinstance(pts_bbox_head, dict) else pts_bbox_head
        )
        self.pts_seg_head = (
            builder.build_head(pts_seg_head)
            if isinstance(pts_seg_head, dict) else pts_seg_head
        )
        self.img_roi_head = (
            builder.build_head(img_roi_head)
            if isinstance(img_roi_head, dict) else img_roi_head
        )
        self.img_rpn_head = (
            builder.build_head(img_rpn_head)
            if isinstance(img_rpn_head, dict) else img_rpn_head
        )
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.pretrained = pretrained
        self.init_cfg = init_cfg

    def init_weights(self) -> None:
        for m in self.children():
            init_fn = getattr(m, "init_weights", None)
            if callable(init_fn):
                init_fn()

    @property
    def with_img_backbone(self) -> bool:
        return self.img_backbone is not None

    @property
    def with_img_neck(self) -> bool:
        return self.img_neck is not None

    @property
    def with_pts_bbox(self) -> bool:
        return self.pts_bbox_head is not None

    @property
    def with_pts_neck(self) -> bool:
        return self.pts_neck is not None

    @property
    def with_pts_seg(self) -> bool:
        return self.pts_seg_head is not None

    def voxelize(self, pts):
        if self.pts_voxel_layer is None:
            raise RuntimeError("pts_voxel_layer is missing.")
        voxels, coors, num_points = [], [], []
        for res in pts:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = torch.nn.functional.pad(coor, (1, 0), mode="constant", value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def extract_pts_feat(self, pts, img_feats, img_metas):
        del img_feats, img_metas
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        if self.pts_voxel_encoder is None or self.pts_middle_encoder is None or self.pts_backbone is None:
            raise RuntimeError("Point branch modules are not fully initialized.")
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            assert self.pts_neck is not None
            x = self.pts_neck(x)
        return x

    def forward_pts_train(
        self,
        pts_feats,
        gt_bboxes_3d,
        gt_labels_3d,
        img_metas,
        gt_bboxes_ignore=None,
    ):
        del img_metas, gt_bboxes_ignore
        if self.pts_bbox_head is None:
            raise RuntimeError("pts_bbox_head is missing.")
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        return self.pts_bbox_head.loss(*loss_inputs)

    def simple_test_pts(self, x, img_metas, rescale: bool = False):
        if self.pts_bbox_head is None:
            raise RuntimeError("pts_bbox_head is missing.")
        outs = self.pts_bbox_head(x)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        return [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

    def aug_test_pts(self, feats, img_metas, rescale: bool = False):
        if self.pts_bbox_head is None:
            raise RuntimeError("pts_bbox_head is missing.")
        outs_list = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.pts_bbox_head(x)
            for task_id, out in enumerate(outs):
                for key in out[0].keys():
                    if img_meta[0]["pcd_horizontal_flip"]:
                        outs[task_id][0][key] = torch.flip(outs[task_id][0][key], dims=[2])
                        if key == "reg":
                            outs[task_id][0][key][:, 1, ...] = 1 - outs[task_id][0][key][:, 1, ...]
                        elif key == "rot":
                            outs[task_id][0][key][:, 1, ...] = -outs[task_id][0][key][:, 1, ...]
                        elif key == "vel":
                            outs[task_id][0][key][:, 1, ...] = -outs[task_id][0][key][:, 1, ...]
                    if img_meta[0]["pcd_vertical_flip"]:
                        outs[task_id][0][key] = torch.flip(outs[task_id][0][key], dims=[3])
                        if key == "reg":
                            outs[task_id][0][key][:, 0, ...] = 1 - outs[task_id][0][key][:, 0, ...]
                        elif key == "rot":
                            outs[task_id][0][key][:, 0, ...] = -outs[task_id][0][key][:, 0, ...]
                        elif key == "vel":
                            outs[task_id][0][key][:, 0, ...] = -outs[task_id][0][key][:, 0, ...]
            outs_list.append(outs)

        # Keep the decoded first-aug path local; full multi-scale merge is a
        # downstream enhancement that previously depended on mmdet3d core ops.
        bbox_list = self.pts_bbox_head.get_bboxes(outs_list[0], img_metas[0], rescale=rescale)
        return [
            dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
            for bboxes, scores, labels in bbox_list
        ][0]

    def aug_test(self, points, img_metas, imgs=None, rescale: bool = False):
        _img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)
        bbox_list = {}
        if pts_feats and self.with_pts_bbox:
            pts_bbox = self.aug_test_pts(pts_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=pts_bbox)
        return [bbox_list]

    def extract_feats(self, points, img_metas, imgs=None):
        """Extract point and image features for augmented samples."""
        if imgs is None:
            imgs = [None] * len(img_metas)

        if not isinstance(points, Sequence):
            raise TypeError("points must be a sequence for augmented testing.")

        img_feats = []
        pts_feats = []
        for pts_i, img_i, img_meta_i in zip(points, imgs, img_metas):
            extract_feat_fn = cast(Any, self).extract_feat
            feat_out = extract_feat_fn(pts_i, img_i, img_meta_i)
            if not isinstance(feat_out, tuple):
                raise TypeError("extract_feat must return a tuple.")
            if len(feat_out) < 2:
                raise ValueError("extract_feat must return at least (img_feats, pts_feats).")
            img_feat_i, pts_feat_i = feat_out[0], feat_out[1]
            img_feats.append(img_feat_i)
            pts_feats.append(pts_feat_i)
        return img_feats, pts_feats
