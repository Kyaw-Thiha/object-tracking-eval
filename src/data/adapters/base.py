"""
Base adapter interface for dataset-specific loaders.
Adapters build an index of frame metadata, then construct `Frame` objects on demand.
"""

from abc import ABC, abstractmethod
from ..schema.frame import Frame, SensorFrame
from ..schema.overlay import OverlaySet


class BaseAdapter(ABC):
    """
    Base class for dataset adapters.

    Subclasses should implement `index_frames`, `load_sensors`, and `load_overlays`.
    `__getitem__` should assemble a `Frame` from indexed metadata.
    """

    dataset_name: str
    dataset_path: str
    frames: list[dict]

    def __init__(self, dataset_name: str, dataset_path: str) -> None:
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.frames = self.index_frames()

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Frame:
        return self.get_frame(idx)

    # Getters
    def get_frame(self, idx: int) -> Frame:
        """
        Build a Frame from the indexed metadata at `frames[idx]`.

        `index_frames` must return dicts with at least:
            - "timestamp": float
            - "frame_id": int

        Any additional keys required by `load_sensors` or `load_overlays`
        should also be present in the dict.
        """
        info = self.frames[idx]
        sensors = self.load_sensors(info)
        overlays = self.load_overlays(info)
        return Frame(
            timestamp=info["timestamp"],
            frame_id=info["frame_id"],
            sensors=sensors,
            overlays=overlays,
            meta=info.get("meta"),
        )

    def get_sequence_ids(self) -> list[str]:
        """Return available sequence ids, or [] if the dataset is not sequenced."""
        return []

    def get_sensor_ids(self) -> list[str]:
        """Return available sensor ids, or [] if not tracked by the adapter."""
        return []

    # Helpers
    @abstractmethod
    def index_frames(self) -> list[dict]:
        """Build and return a list of per-frame metadata dicts used by __getitem__."""

    @abstractmethod
    def load_sensors(self, info: dict) -> dict[str, SensorFrame]:
        """Load sensor data for a frame and return a sensor_id -> SensorFrame mapping."""

    @abstractmethod
    def load_overlays(self, info: dict) -> OverlaySet | None:
        """Load overlays/labels for a frame; return None if unlabeled."""
