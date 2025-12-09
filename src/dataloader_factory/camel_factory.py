from datasets.camel_dataset import CAMELCocoDataset
import torch
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset


def factory(batch_size=2, num_workers=0, input_size=(640, 640)):
    # --- Build dataset + dataloader ---
    ann_file_path_test = '/home/allynbao/project/UncertaintyTrack/src/data/camel_dataset/annotations/test_cocoformat.json'
    # test subdirectory path
    dataset_dir = '/home/allynbao/project/UncertaintyTrack/src/data/camel_dataset/'

    # test_dataset = CAMELDataset(dataset_dir, input_size=input_size)
    test_dataset = CAMELCocoDataset(ann_file_path_test, dataset_dir, input_size)
    # combined_dataset = ConcatDataset([train_dataset, val_dataset])
    def detection_collate_fn(batch):
        images = torch.stack([b[0] for b in batch])      # stack image tensors
        targets = [b[1] for b in batch]                  # keep list of dicts/arrays
        return images, targets
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=detection_collate_fn)

    return dataloader