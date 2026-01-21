from typing import Any
import numpy as np
import torch
import cv2 as cv

from ..schema.image import ImageSensorFrame
from ..schema.frame import Frame
from ..schema.overlay import Box2D
from .base import BaseProducer


class CamelEvaluationProducer(BaseProducer):
    def __init__(self, adapter, input_size: tuple[int, int] = (640, 640)) -> None:
        super().__init__(adapter)
        self.input_size = input_size  # (H, W) to match CAMELCocoDataset letterbox

    def get_sensors(self, frame: Frame) -> dict[str, Any]:
        # Single-camera input for CAMEL
        data = frame.sensors["cam"].data
        img = {}
        if isinstance(data, ImageSensorFrame):
            img = data.image  # BGR from cv2.imread
        return {"cam": img}

    def preprocess(self, sensors: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        img = sensors["cam"]
        if img is not None:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_resized, scale, pad_shape = self.letterbox(img, self.input_size)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
        return {"cam": img_tensor}, {"scale_factor": scale, "ori_shape": img.shape, "pad_shape": pad_shape}

    def build_target(self, frame: Frame, meta: dict[str, Any]) -> dict[str, Any]:
        boxes_list: list[list[float]] = []
        labels_list: list[int] = []

        overlays = frame.overlays
        if overlays and "gt" in overlays.boxes2D:
            for box in overlays.boxes2D["gt"]:
                assert isinstance(box, Box2D)
                x1, y1, x2, y2 = box.xyxy.tolist()
                boxes_list.append([x1, y1, x2, y2])
                labels_list.append(int(box.class_id))

        boxes = torch.tensor(boxes_list, dtype=torch.float32) if boxes_list else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels_list, dtype=torch.long) if labels_list else torch.zeros((0,), dtype=torch.long)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([frame.frame_id]),
            "video_id": frame.meta.sequence_id if frame.meta else "",
            "frame_id": frame.frame_id,
            "img_metas": {
                "scale_factor": meta["scale_factor"],
                "ori_shape": meta["ori_shape"],
                "img_shape": meta["pad_shape"],
                "pad_shape": meta["pad_shape"],
            },
        }
        return target

    def letterbox(self, img: np.ndarray, target_size: tuple[int, int]) -> tuple[np.ndarray, float, tuple[int, int, int]]:
        target_h, target_w = target_size

        # Resizing (match CAMELCocoDataset behavior)
        ratio = min(target_h / img.shape[0], target_w / img.shape[1])
        resized_w = int(img.shape[1] * ratio)
        resized_h = int(img.shape[0] * ratio)
        resized = cv.resize(img, (resized_w, resized_h), interpolation=cv.INTER_LINEAR).astype(np.float32)

        # Fixed-size padding to target
        padded = np.ones((target_h, target_w, 3), dtype=np.float32) * 114.0
        padded[: resized.shape[0], : resized.shape[1]] = resized
        return padded, ratio, (target_h, target_w, 3)
