"""RCBEVDet package."""

from .registry import (
    BACKBONES,
    DETECTORS,
    HEADS,
    MIDDLE_ENCODERS,
    NECKS,
    VOXEL_ENCODERS,
    build_backbone,
    build_detector,
    build_head,
    build_middle_encoder,
    build_neck,
    build_voxel_encoder,
)
from .backbones import CustomResNet, CustomResNet3D, Down2TopResNet, RadarBEVNet, SECOND
from .detectors import BEVDepth4D_RC, BEVDepth4D_RC_d2t, BEVDet4D_RC, BEVDet_RC, BEVStereo4D_RC, CenterPoint
from .heads import CenterHead, CenterHeadkitti, DCNSeparateHead, SeparateHead
from .core.centerpoint_bbox_coders import CenterPointBBoxCoder
from .middle_encoders import PointPillarsScatter, PointPillarsScatterRCS
from .necks import (
    CustomSECONDFPN,
    FPN_LSS,
    LSSFPN3D,
    LSSViewTransformer,
    LSSViewTransformerBEVDepth,
    LSSViewTransformerBEVStereo,
    LSSViewTransformerVOD,
    SECONDFPN,
)


def _register_detectors_to_mmdet() -> None:
    """Mirror RCBEVDet detector classes into MMDet registry for compatibility."""
    try:
        from mmdet.models.builder import DETECTORS as MMDET_DETECTORS
    except Exception:
        return

    detector_classes = [
        BEVDet_RC,
        BEVDet4D_RC,
        BEVDepth4D_RC,
        BEVDepth4D_RC_d2t,
        BEVStereo4D_RC,
    ]
    for cls in detector_classes:
        if cls.__name__ not in MMDET_DETECTORS.module_dict:
            MMDET_DETECTORS.register_module()(cls)


_register_detectors_to_mmdet()

__all__ = [
    "BACKBONES",
    "NECKS",
    "MIDDLE_ENCODERS",
    "VOXEL_ENCODERS",
    "DETECTORS",
    "HEADS",
    "build_backbone",
    "build_neck",
    "build_middle_encoder",
    "build_voxel_encoder",
    "build_detector",
    "build_head",
    "CustomResNet",
    "Down2TopResNet",
    "CustomResNet3D",
    "SECOND",
    "RadarBEVNet",
    "CenterPoint",
    "BEVDet_RC",
    "BEVDet4D_RC",
    "BEVDepth4D_RC",
    "BEVDepth4D_RC_d2t",
    "BEVStereo4D_RC",
    "SeparateHead",
    "DCNSeparateHead",
    "CenterHead",
    "CenterHeadkitti",
    "CenterPointBBoxCoder",
    "PointPillarsScatter",
    "PointPillarsScatterRCS",
    "FPN_LSS",
    "LSSFPN3D",
    "SECONDFPN",
    "CustomSECONDFPN",
    "LSSViewTransformer",
    "LSSViewTransformerBEVDepth",
    "LSSViewTransformerBEVStereo",
    "LSSViewTransformerVOD",
]
