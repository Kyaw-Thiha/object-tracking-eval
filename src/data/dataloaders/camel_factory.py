import torch
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from pathlib import Path

from data.datasets.camel_dataset import CAMELCocoDataset
from data.adapters.camel import CamelAdapter
from data.producers.camel import CamelEvaluationProducer

SRC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SRC_ROOT.parent


def factory(batch_size=2, num_workers=0, input_size=(336, 256)):
    dataset_path = PROJECT_ROOT / "data" / "camel_dataset"
    adapter = CamelAdapter(dataset_path=str(dataset_path), ann_file="annotations/test_cocoformat_half.json", split="test_half")
    dataset = CamelEvaluationProducer(adapter, input_size=input_size)

    def detection_collate_fn(batch):
        images = torch.stack([b[0]["cam"] for b in batch])
        targets = [b[1] for b in batch]
        return images, targets

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=detection_collate_fn)


# def factory(batch_size=2, num_workers=0, input_size=(336, 256)):
#     # --- Build dataset + dataloader ---
#     dataset_dir = PROJECT_ROOT / "data" / "camel_dataset"
#     # ann_file_path_test = dataset_dir / "annotations" / "test_cocoformat.json"
#     ann_file_path_test = dataset_dir / "annotations" / "test_cocoformat_half.json"
#
#     # test_dataset = CAMELDataset(dataset_dir, input_size=input_size)
#     test_dataset = CAMELCocoDataset(str(ann_file_path_test), str(dataset_dir), input_size)
#
#     # combined_dataset = ConcatDataset([train_dataset, val_dataset])
#     def detection_collate_fn(batch):
#         images = torch.stack([b[0] for b in batch])  # stack image tensors
#         targets = [b[1] for b in batch]  # keep list of dicts/arrays
#         return images, targets
#
#     dataloader = DataLoader(
#         test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=detection_collate_fn
#     )
#
#     return dataloader
