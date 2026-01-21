from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import Dataset

from ..adapters.base import BaseAdapter
from ..schema.frame import Frame


class BaseProducer(Dataset, ABC):
    def __init__(self, adapter: BaseAdapter) -> None:
        self.adapter = adapter

    def __len__(self) -> int:
        return len(self.adapter)

    def __getitem__(self, idx: int) -> tuple[dict[str, Any], dict[str, Any]]:
        frame = self.adapter[idx]
        sensors = self.get_sensors(frame)
        sensors, meta = self.preprocess(sensors)
        target = self.build_target(frame, meta)
        return sensors, target

    @abstractmethod
    def get_sensors(self, frame: Frame) -> dict[str, Any]: ...

    @abstractmethod
    def preprocess(self, sensors: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        # Default: no-op
        return sensors, {}

    @abstractmethod
    def build_target(self, frame: Frame, meta: dict[str, Any]) -> dict[str, Any]: ...
