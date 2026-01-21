from typing import Any, cast
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO

from ..schema.frame import FrameMeta, SensorFrame
from ..schema.image import ImageMeta, ImageSensorFrame
from ..schema.overlay import Box2D, OverlayMeta, OverlaySet
from .base import BaseAdapter


class CocoBaseAdapter(BaseAdapter):
    split: str
    ann_path: Path
    images_root: Path

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str,
        ann_file: str,
        split: str,
        images_root: str | None = None,
    ) -> None:
        # Basic dataset config
        self.split = split
        self.ann_path = Path(dataset_path) / ann_file
        self.images_root = Path(images_root) if images_root else Path(dataset_path) / split

        # COCO API + contiguous class ids
        self.coco = COCO(str(self.ann_path))
        categories = sorted(self.coco.dataset.get("categories", []), key=lambda c: c["id"])
        self.category_id_to_class = {category["id"]: idx for idx, category in enumerate(categories)}

        super().__init__(dataset_name, dataset_path)

    # Hooks to specialize per dataset if needed
    def get_sequence_id(self, img: Any) -> str | None:
        img = cast(dict[str, Any], img)
        return img.get("video_id")

    def get_frame_id(self, img: Any) -> int:
        img = cast(dict[str, Any], img)
        frame_id = img.get("frame_id")
        if frame_id is None:
            frame_id = img.get("mot_frame_id", img["id"])
        return int(frame_id)

    def get_timestamp(self, img: Any) -> float:
        return float(self.get_frame_id(img))

    def get_image_path(self, img: Any) -> Path:
        img = cast(dict[str, Any], img)
        return self.images_root / img["file_name"]

    def map_category_id(self, category_id: int) -> int | None:
        return self.category_id_to_class.get(category_id)

    def filter_annotation(self, ann: Any) -> bool:
        # Filter crowds by default
        ann = cast(dict[str, Any], ann)
        return not ann.get("iscrowd", False)

    def get_track_id(self, ann: Any) -> int | None:
        ann = cast(dict[str, Any], ann)
        return ann.get("track_id")

    def get_sequence_ids(self) -> list[str]:
        video_ids = {img.get("video_id") for img in self.coco.imgs.values()}
        return sorted([vid for vid in video_ids if vid is not None])

    def get_sensor_ids(self) -> list[str]:
        # COCO is image-only for our use case
        return ["cam"]

    def index_frames(self) -> list[dict]:
        def sort_key(img: dict[str, Any]) -> tuple[str, int]:
            # Ensure a deterministic ordering (sequence -> frame)
            seq = str(self.get_sequence_id(img) or "")
            return (seq, self.get_frame_id(img))

        frames = []
        images = cast(list[dict[str, Any]], list(self.coco.imgs.values()))
        for img in sorted(images, key=sort_key):
            frames.append(
                {
                    "image_id": img["id"],
                    "timestamp": self.get_timestamp(img),
                    "frame_id": self.get_frame_id(img),
                    "meta": FrameMeta(
                        sequence_id=self.get_sequence_id(img),
                        dataset=self.dataset_name,
                        scene=None,
                        split=self.split,
                        weather=None,
                    ),
                }
            )
        return frames

    def load_sensors(self, info: dict) -> dict[str, SensorFrame]:
        # Load the image
        img_info = self.coco.imgs[info["image_id"]]
        image_path = self.get_image_path(img_info)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Default intrinsics/pose (COCO doesn't provide camera params)
        intrinsics = np.eye(3, dtype=np.float32)
        sensor_pose_in_ego = np.eye(4, dtype=np.float32)

        meta = ImageMeta(
            spectral="rgb",
            frame="sensor:cam",
            intrinsics=intrinsics,
            sensor_pose_in_ego=sensor_pose_in_ego,
        )
        data = ImageSensorFrame(sensor_id="cam", meta=meta, image=image, mask=None)
        return {"cam": SensorFrame(kind="image", data=data)}

    def load_overlays(self, info: dict) -> OverlaySet | None:
        ann_ids = self.coco.getAnnIds(imgIds=[info["image_id"]])
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        for ann in anns:
            # Filter out unwanted annotations
            if not self.filter_annotation(ann):
                continue

            # Map category ids to contiguous class ids
            class_id = self.map_category_id(ann["category_id"])
            if class_id is None:
                continue

            # Convert xywh -> xyxy
            x, y, w, h = ann["bbox"]
            xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)

            meta = OverlayMeta(
                coord_frame="sensor:cam",
                source="gt",
                timestamp=info["timestamp"],
                sensor_id="cam",
            )
            boxes.append(
                Box2D(
                    meta=meta,
                    xyxy=xyxy,
                    class_id=class_id,
                    confidence=None,
                    track_id=self.get_track_id(ann),
                )
            )

        return OverlaySet(
            boxes3D={},
            boxes2D={"gt": boxes},
            oriented_boxes2d={},
            radar_dets={},
            radar_polar_dets={},
            tracks={},
        )
