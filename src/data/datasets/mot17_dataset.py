import os
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import numpy as np
import cv2

class MOT17CocoDataset(Dataset):
    def __init__(self, ann_file, img_prefix, input_size=(640, 640)):
        self.coco = COCO(ann_file)
        self.img_prefix = img_prefix
        self.input_size = input_size

        # group images by video (MOT17 uses video sequence in filename prefix)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cat_id_to_label = {cat_id: i for i, cat_id in enumerate(self.coco.getCatIds())}

        # keep mapping: image_id -> video_name, frame_id
        self.frame_info = {}
        for img_id in self.ids:
            img_info = self.coco.loadImgs(img_id)[0]
            filename = img_info['file_name']  # e.g. MOT17-03/img1/000001.jpg
            video_id = filename.split("/")[0]  # "MOT17-03"
            frame_id = int(os.path.splitext(filename.split("/")[-1])[0])  # 000001 -> 1
            self.frame_info[img_id] = dict(video_id=video_id, frame_id=frame_id)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # --- Load image ---
        img_info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_prefix, img_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- GT boxes ---
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])  # xyxy
            labels.append(self.cat_id_to_label[ann['category_id']])

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "video_id": self.frame_info[img_id]["video_id"],
            "frame_id": self.frame_info[img_id]["frame_id"],
        }

        img_tensor = self.preprocess(img)

        return img_tensor, target

    def preprocess(self, img):
        h, w = img.shape[:2]
        ih, iw = self.input_size
        scale = min(ih / h, iw / w)
        nh, nw = int(h * scale), int(w * scale)

        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((ih, iw, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw, :] = resized

        tensor = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
        return tensor