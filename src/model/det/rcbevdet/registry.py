"""Local-first model builders for RCBEVDet with MM fallback support."""
from __future__ import annotations

from typing import Any

from mmcv.utils import Registry

BACKBONES = Registry('backbone')
NECKS = Registry('neck')
MIDDLE_ENCODERS = Registry('middle_encoder')
VOXEL_ENCODERS = Registry('voxel_encoder')
DETECTORS = Registry('detector')
HEADS = Registry('head')


def _import_mm_builders() -> tuple[Any, Any]:
    """Lazily import external MM builders to avoid hard import at module load."""
    try:
        from mmdet.models import builder as mmdet_builder
    except Exception:
        mmdet_builder = None
    try:
        from mmdet3d.models import builder as mmdet3d_builder
    except Exception:
        mmdet3d_builder = None
    return mmdet_builder, mmdet3d_builder


def _build_local_or_fallback(
    registry: Registry,
    cfg: dict[str, Any],
    mmdet_fn: str | None = None,
    mmdet3d_fn: str | None = None,
):
    module_type = cfg.get("type")
    if module_type in registry.module_dict:
        return registry.build(cfg)

    mmdet_builder, mmdet3d_builder = _import_mm_builders()
    if mmdet3d_fn is not None and mmdet3d_builder is not None:
        fn = getattr(mmdet3d_builder, mmdet3d_fn, None)
        if fn is not None:
            return fn(cfg)
    if mmdet_fn is not None and mmdet_builder is not None:
        fn = getattr(mmdet_builder, mmdet_fn, None)
        if fn is not None:
            return fn(cfg)

    raise KeyError(
        f"Unable to build module type '{module_type}'. "
        "Not found in local RCBEVDet registry and no compatible MM fallback available."
    )


def build_backbone(cfg):
    """Build backbone from config dict."""
    return _build_local_or_fallback(
        BACKBONES,
        cfg,
        mmdet_fn="build_backbone",
        mmdet3d_fn="build_backbone",
    )

def build_neck(cfg):
    """Build neck from config dict."""
    return _build_local_or_fallback(
        NECKS,
        cfg,
        mmdet_fn="build_neck",
        mmdet3d_fn="build_neck",
    )

def build_middle_encoder(cfg):
    """Build middle encoder from config dict."""
    return _build_local_or_fallback(
        MIDDLE_ENCODERS,
        cfg,
        mmdet3d_fn="build_middle_encoder",
    )

def build_voxel_encoder(cfg):
    """Build voxel encoder from config dict."""
    return _build_local_or_fallback(
        VOXEL_ENCODERS,
        cfg,
        mmdet3d_fn="build_voxel_encoder",
    )

def build_detector(cfg):
    """Build detector from config dict."""
    return _build_local_or_fallback(
        DETECTORS,
        cfg,
        mmdet_fn="build_detector",
        mmdet3d_fn="build_detector",
    )

def build_head(cfg):
    """Build head from config dict."""
    return _build_local_or_fallback(
        HEADS,
        cfg,
        mmdet_fn="build_head",
        mmdet3d_fn="build_head",
    )
