"""Base view interfaces for RenderSpec builders."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from ..schema.render_spec import RenderSpec, RenderSpecMeta
from ...data.schema.frame import Frame

ConfigType = TypeVar("ConfigType")


class BaseView(ABC, Generic[ConfigType]):
    """Base class for view builders that produce RenderSpec objects."""

    name: str

    @abstractmethod
    def build(self, frame: Frame, cfg: ConfigType) -> RenderSpec:
        """Build a RenderSpec for a single frame using the provided config."""
        ...

    def build_meta(self, frame: Frame, sensor_ids: list[str], source_keys: list[str]) -> RenderSpecMeta:
        """Build common RenderSpec metadata for a view."""
        return RenderSpecMeta(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            dataset=frame.meta.dataset if frame.meta else None,
            sequence_id=frame.meta.sequence_id if frame.meta else None,
            scene=frame.meta.scene if frame.meta else None,
            split=frame.meta.split if frame.meta else None,
            weather=frame.meta.weather if frame.meta else None,
            view_name=self.name,
            sensor_ids=sensor_ids,
            source_keys=source_keys,
        )
