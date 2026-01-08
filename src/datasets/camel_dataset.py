import os
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import numpy as np
import cv2


class CAMELCocoDataset(Dataset):
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
            # camel COCO stores video_id and frame ids in the JSON
            video_id = img_info.get("video_id")
            frame_id = img_info.get("mot_frame_id", img_info.get("frame_id"))
            self.frame_info[img_id] = dict(video_id=video_id, frame_id=frame_id)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # --- Load image ---
        img_info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_prefix, img_info["file_name"])
        img = cv2.imread(path)

        # --- GT boxes ---
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])  # xyxy
            labels.append(self.cat_id_to_label[ann["category_id"]])

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long)

        img_tensor, scale, pad_shape = self.preprocess(img)
        ih, iw = pad_shape
        img_meta = {
            "ori_shape": img.shape,
            "img_shape": (ih, iw, 3),
            "pad_shape": (ih, iw, 3),
            "scale_factor": np.array([scale, scale, scale, scale], dtype=np.float32),
            "flip": False,
            "flip_direction": None,
            "ori_filename": img_info["file_name"],
        }

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "video_id": self.frame_info[img_id]["video_id"],
            "frame_id": self.frame_info[img_id]["frame_id"],
            "img_metas": img_meta,
        }

        return img_tensor, target

    def preprocess(self, img):
        h, w = img.shape[:2]
        ih, iw = self.input_size
        scale = min(ih / h, iw / w)
        nh, nw = int(h * scale), int(w * scale)

        resized = cv2.resize(img, (nw, nh))
        pad_h = int(np.ceil(nh / 32) * 32)
        pad_w = int(np.ceil(nw / 32) * 32)
        canvas = np.full((pad_h, pad_w, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw, :] = resized

        tensor = torch.from_numpy(canvas).permute(2, 0, 1).float()
        return tensor, scale, (pad_h, pad_w)
