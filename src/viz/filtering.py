"""Filtering utilities for overlays and render specs."""

from __future__ import annotations

from .schema.render_spec import RenderSpec
from ..data.schema.overlay import OverlaySet, Box2D, Box3D, RadarPointDetections, RadarPolarDetections, Track


def filter_overlay_boxes2d(
    overlays: OverlaySet | None,
    source_key: str,
    sensor_id: str | None = None,
    coord_frame: str | None = None,
) -> list[Box2D]:
    """Return 2D boxes filtered by source, sensor, and coordinate frame."""
    if overlays is None or source_key not in overlays.boxes2D:
        return []

    out: list[Box2D] = []
    for box in overlays.boxes2D[source_key]:
        if sensor_id is not None and box.meta.sensor_id != sensor_id:
            continue
        if coord_frame is not None and box.meta.coord_frame != coord_frame:
            continue
        out.append(box)
    return out


def filter_overlay_boxes3d(
    overlays: OverlaySet | None,
    source_key: str,
    coord_frame: str | None = None,
) -> list[Box3D]:
    """Return 3D boxes filtered by source and coordinate frame."""
    if overlays is None or source_key not in overlays.boxes3D:
        return []

    out: list[Box3D] = []
    for box in overlays.boxes3D[source_key]:
        if coord_frame is not None and box.meta.coord_frame != coord_frame:
            continue
        out.append(box)
    return out


def filter_overlay_tracks(
    overlays: OverlaySet | None,
    source_key: str,
    sensor_id: str | None = None,
    coord_frame: str | None = None,
) -> list[Track]:
    """Return tracks filtered by source, sensor, and coordinate frame."""
    if overlays is None or source_key not in overlays.tracks:
        return []

    out: list[Track] = []
    for track in overlays.tracks[source_key]:
        if sensor_id is not None and track.meta.sensor_id != sensor_id:
            continue
        if coord_frame is not None and track.meta.coord_frame != coord_frame:
            continue
        out.append(track)
    return out


def filter_overlay_radar_dets(
    overlays: OverlaySet | None,
    source_key: str,
    sensor_id: str | None = None,
    coord_frame: str | None = None,
) -> RadarPointDetections | None:
    """Return radar point detections filtered by source, sensor, and coordinate frame."""
    if overlays is None or source_key not in overlays.radar_dets:
        return None

    dets = overlays.radar_dets[source_key]
    if sensor_id is not None and dets.meta.sensor_id != sensor_id:
        return None
    if coord_frame is not None and dets.meta.coord_frame != coord_frame:
        return None
    return dets


def filter_overlay_radar_polar_dets(
    overlays: OverlaySet | None,
    source_key: str,
    sensor_id: str | None = None,
    coord_frame: str | None = None,
) -> RadarPolarDetections | None:
    """Return radar polar detections filtered by source, sensor, and coordinate frame."""
    if overlays is None or source_key not in overlays.radar_polar_dets:
        return None

    dets = overlays.radar_polar_dets[source_key]
    if sensor_id is not None and dets.meta.sensor_id != sensor_id:
        return None
    if coord_frame is not None and dets.meta.coord_frame != coord_frame:
        return None
    return dets


def filter_layers_by_meta(
    spec: RenderSpec,
    source: str | None = None,
    sensor_id: str | None = None,
    kind: str | None = None,
) -> RenderSpec:
    """Return a shallow copy of RenderSpec with layers filtered by LayerMeta."""
    filtered = []
    for layer in spec.layers:
        meta = layer.meta
        if source is not None and meta.source != source:
            continue
        if sensor_id is not None and meta.sensor_id != sensor_id:
            continue
        if kind is not None and meta.kind != kind:
            continue
        filtered.append(layer)

    return RenderSpec(
        title=spec.title,
        coord_frame=spec.coord_frame,
        layers=filtered,
        meta=spec.meta,
    )
