from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data.adapters.nuscenes_rc import NuScenesRCAdapter
from data.producers.rcbevdet import RCBEVDetProducer

SRC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SRC_ROOT.parent


def rcbevdet_collate_fn(batch: list[tuple[dict[str, Any], dict[str, Any]]]):
    """Collate RCBEVDet batch while preserving variable-size radar inputs."""
    images = torch.stack([item[0]["images"] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets


def factory(
    batch_size: int = 1,
    num_workers: int = 0,
    dataset_path: str | None = None,
    camera_ids: list[str] | None = None,
    radar_ids: list[str] | None = None,
    num_sweeps: int = 9,
    input_size: tuple[int, int] = (256, 704),
):
    """NuScenes RCBEVDet dataloader factory."""
    if dataset_path is None:
        dataset_path = str(PROJECT_ROOT / "data" / "nuScenes")

    adapter = NuScenesRCAdapter(dataset_path=dataset_path)
    producer = RCBEVDetProducer(
        adapter=adapter,
        camera_ids=camera_ids,
        radar_ids=radar_ids,
        num_sweeps=num_sweeps,
        input_size=input_size,
    )

    return DataLoader(
        producer,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=rcbevdet_collate_fn,
    )
