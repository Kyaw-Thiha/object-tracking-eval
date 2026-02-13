"""Core utility modules for RCBEVDet."""

from .nn_utils import (
    BasicBlock,
    Bottleneck,
    ConvModule,
    build_norm_layer,
    force_fp32,
    normal_init,
)
from .gaussian import draw_heatmap_gaussian, draw_heatmap_gaussian_feat
from .clip_sigmoid import clip_sigmoid
from .bbox_utils import xywhr2xyxyr
from .box3d_nms import circle_nms, nms_bev, oval_nms

# Register CenterPoint bbox coder into mmdet bbox coder registry.
from .centerpoint_bbox_coders import CenterPointBBoxCoder  # noqa: F401

__all__ = [
    "BasicBlock",
    "Bottleneck",
    "ConvModule",
    "build_norm_layer",
    "force_fp32",
    "normal_init",
    "draw_heatmap_gaussian",
    "draw_heatmap_gaussian_feat",
    "clip_sigmoid",
    "xywhr2xyxyr",
    "circle_nms",
    "oval_nms",
    "nms_bev",
]
