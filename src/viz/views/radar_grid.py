"""Radar grid view builder."""

from dataclasses import dataclass
from typing import Literal

from ...data.schema.radar import RadarSensorFrame

from .base import BaseView
from ..schema.render_spec import RenderSpec
from ..schema.layers import RasterLayer
from ..schema.base_layer import Layer, LayerMeta
from ...data.schema.frame import Frame


@dataclass
class RadarGridViewConfig:
    """Configuration for RadarGridView rendering."""

    sensor_id: str
    source_key: str
    grid_name: str
    display: Literal["pixel", "polar"] = "pixel"


class RadarGridView(BaseView[RadarGridViewConfig]):
    """Builds a radar grid RenderSpec from RAD/RA/RD tensors."""

    name = "RadarGridView"

    def build(self, frame: Frame, cfg: RadarGridViewConfig) -> RenderSpec:
        """Assemble the radar grid view layers for a frame."""
        layers: list[Layer] = [self.build_grid_layer(frame, cfg)]
        meta = self.build_meta(frame, [cfg.sensor_id], [cfg.source_key])

        radar = frame.sensors[cfg.sensor_id].data
        return RenderSpec(title=f"{cfg.sensor_id}:{cfg.grid_name}", coord_frame=radar.meta.frame, layers=layers, meta=meta)

    def build_grid_layer(self, frame: Frame, cfg: RadarGridViewConfig) -> RasterLayer:
        """Create the radar grid raster layer for a single grid product."""
        radar = frame.sensors[cfg.sensor_id].data
        assert radar is RadarSensorFrame
        if radar.grids is None or cfg.grid_name not in radar.grids:
            raise ValueError(f"Grid {cfg.grid_name} not found for {cfg.sensor_id}")
        grid = radar.grids[cfg.grid_name]

        return RasterLayer(
            name=f"{cfg.sensor_id}.{cfg.grid_name}",
            meta=LayerMeta(
                source=cfg.source_key,
                sensor_id=cfg.sensor_id,
                kind="grid",
                coord_frame=radar.meta.frame,
                timestamp=frame.timestamp,
            ),
            data=grid.tensor,
            axes=grid.axes,
            bins=grid.bins,
            extent=None,
            grid_name=cfg.grid_name,
            display=cfg.display,
        )
