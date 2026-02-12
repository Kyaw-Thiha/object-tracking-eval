"""Minimal model registry for RCBEVDet components."""
from mmcv.utils import Registry

BACKBONES = Registry('backbone')
NECKS = Registry('neck')
MIDDLE_ENCODERS = Registry('middle_encoder')
VOXEL_ENCODERS = Registry('voxel_encoder')
DETECTORS = Registry('detector')
HEADS = Registry('head')

def build_backbone(cfg):
    """Build backbone from config dict."""
    return BACKBONES.build(cfg)

def build_neck(cfg):
    """Build neck from config dict."""
    return NECKS.build(cfg)

def build_middle_encoder(cfg):
    """Build middle encoder from config dict."""
    return MIDDLE_ENCODERS.build(cfg)

def build_voxel_encoder(cfg):
    """Build voxel encoder from config dict."""
    return VOXEL_ENCODERS.build(cfg)

def build_detector(cfg):
    """Build detector from config dict."""
    return DETECTORS.build(cfg)

def build_head(cfg):
    """Build head from config dict."""
    return HEADS.build(cfg)
