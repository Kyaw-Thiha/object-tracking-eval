import os
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import numpy as np
import cv2 as cv


class CAMELCocoDataset(Dataset):
    def __init__(self, ann_file, img_prefix, input_size=(640, 640)):
        self.coco = COCO(ann_file)
        self.img_prefix = img_prefix
        self.input_size = input_size

        # Image IDs
        self.ids = list(sorted(self.coco.imgs.keys()))

        # Consistent category mapping (same as MOT17 logic)
        self.cat_id_to_label = {cat_id: i for i, cat_id in enumerate(self.coco.getCatIds())}

        # image_id -> (video_id, frame_id)
        self.frame_info = {}
        for img_id in self.ids:
            img_info = self.coco.loadImgs(img_id)[0]
            video_id = img_info["video_id"]
            frame_id = img_info["mot_frame_id"]  # use CAMEL frame index

            self.frame_info[img_id] = dict(video_id=video_id, frame_id=frame_id)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # ---- Load Image ----
        img_info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_prefix, img_info["file_name"])
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # ---- Load GT boxes ----
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])  # convert to xyxy format
            labels.append(self.cat_id_to_label[ann["category_id"]])

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long)

        # ---- Preprocessing (letterbox) ----
        rescaled_img, scale = self.preprocess(img)
        img_tensor = torch.from_numpy(rescaled_img).permute(2, 0, 1).float()

        # ---- Scale GT boxes to resized image coords ----
        if len(boxes) > 0:
            boxes = np.array(boxes, dtype=np.float32)
            boxes *= scale
            boxes = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        # ---- Match MOT17 Dataset Target Format ----
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "video_id": self.frame_info[img_id]["video_id"],
            "frame_id": self.frame_info[img_id]["frame_id"],
            "img_metas": {
                "filename": path,
                "ori_shape": (img_info["height"], img_info["width"], 3),
                "img_shape": (self.input_size[0], self.input_size[1], 3),
                "pad_shape": (self.input_size[0], self.input_size[1], 3),
                "scale_factor": scale,
            },
        }

        return img_tensor, target

    def preprocess(self, img):
        return self.letterbox(img)

    def letterbox(self, img, target_size=(640, 640)):
        h, w = int(target_size[0]), int(target_size[1])
        padded_img = np.ones((h, w, 3), dtype=np.float32) * 114.0

        ratio = min(h / img.shape[0], w / img.shape[1])
        resized_img = cv.resize(
            img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)), interpolation=cv.INTER_LINEAR
        ).astype(np.float32)

        padded_img[: resized_img.shape[0], : resized_img.shape[1]] = resized_img
        return padded_img, ratio
