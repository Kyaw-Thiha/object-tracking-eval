from data.datasets.mot17_dataset import MOT17CocoDataset
import torch
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent


def factory(batch_size=4, num_workers=0, input_size=(640, 640)):
    # --- Build MOT17 dataset + dataloader ---
    mot17_root = PROJECT_ROOT / "data" / "MOT17"
    ann_file_path_train = mot17_root / "annotations" / "half-train_cocoformat.json"
    ann_file_path_val = mot17_root / "annotations" / "half-val_cocoformat.json"
    
    # ann_file_path = mot17_root / "annotations" / "test_cocoformat.json"
    image_prefix_path_train = mot17_root / "train"
    # image_prefix_path_test = mot17_root / "test"
    train_dataset = MOT17CocoDataset(str(ann_file_path_train), str(image_prefix_path_train), input_size=input_size)
    val_dataset = MOT17CocoDataset(str(ann_file_path_val), str(image_prefix_path_train), input_size=input_size)
    # combined_dataset = ConcatDataset([train_dataset, val_dataset])
    def detection_collate_fn(batch):
        images = torch.stack([b[0] for b in batch])      # stack image tensors
        targets = [b[1] for b in batch]                  # keep list of dicts/arrays
        return images, targets
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=detection_collate_fn)
    # combined_loader = DataLoader(
    #     combined_dataset, 
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=num_workers,
    #     collate_fn=detection_collate_fn
    # )
    return dataloader
