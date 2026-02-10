from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.adapters.nuscenes import NuScenesAdapter
from data.producers.rrpn import RRPNProducer

SRC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SRC_ROOT.parent


def factory(
    batch_size: int = 2,
    num_workers: int = 0,
    input_size: tuple[int, int] = (800, 1600),
    dataset_path: str | None = None,
    camera_id: str = "CAM_FRONT",
    radar_ids: list[str] | None = None,
    alpha: float = 20.0,
    beta: float = 0.5,
):
    """
    NuScenes + RRPN dataloader factory.

    Composition:
      NuScenesAdapter (dataset-specific) -> RRPNProducer (task-specific) -> DataLoader
    """
    if dataset_path is None:
        dataset_path = str(PROJECT_ROOT / "data" / "nuScenes")

    adapter = NuScenesAdapter(dataset_path=dataset_path)
    producer = RRPNProducer(
        adapter=adapter,
        input_size=input_size,
        camera_id=camera_id,
        radar_ids=radar_ids,
        alpha=alpha,
        beta=beta,
    )

    def detection_collate_fn(batch):
        # batch item = (sensors_dict, target_dict)
        images = torch.stack([item[0]["image"] for item in batch])
        targets = [item[1] for item in batch]
        return images, targets

    return DataLoader(
        producer,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
    )
