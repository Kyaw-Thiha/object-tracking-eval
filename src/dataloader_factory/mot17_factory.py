from datasets.mot17_dataset import MOT17CocoDataset
import torch
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset


def factory(batch_size=4, num_workers=0, input_size=(640, 640)):
    # --- Build MOT17 dataset + dataloader ---
    ann_file_path_train = '/home/allynbao/project/UncertaintyTrack/src/data/MOT17/annotations/half-train_cocoformat.json'
    ann_file_path_val = '/home/allynbao/project/UncertaintyTrack/src/data/MOT17/annotations/half-val_cocoformat.json'
    
    # ann_file_path = '/home/allynbao/project/UncertaintyTrack/src/data/MOT17/annotations/test_cocoformat.json'
    image_prefix_path_train = '/home/allynbao/project/UncertaintyTrack/src/data/MOT17/train'
    # image_prefix_path_test = '/home/allynbao/project/UncertaintyTrack/src/data/MOT17/test'
    train_dataset = MOT17CocoDataset(ann_file_path_train, image_prefix_path_train, input_size=input_size)
    val_dataset = MOT17CocoDataset(ann_file_path_val, image_prefix_path_train, input_size=input_size)
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