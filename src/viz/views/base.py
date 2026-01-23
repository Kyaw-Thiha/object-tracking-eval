from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from ..schema.render_spec import RenderSpec, RenderSpecMeta
from ...data.schema.frame import Frame

ConfigType = TypeVar("ConfigType")


class BaseView(ABC, Generic[ConfigType]):
    name: str

    @abstractmethod
    def build(self, frame: Frame, cfg: ConfigType) -> RenderSpec: ...

    def build_meta(self, frame: Frame, sensor_ids: list[str], source_keys: list[str]) -> RenderSpecMeta:
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
