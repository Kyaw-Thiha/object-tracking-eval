from typing import Any, cast
import cv2
import numpy as np

from pycocotools.coco import COCO

from ..schema.image import ImageMeta, ImageSensorFrame
from ..schema.frame import FrameMeta, SensorFrame
from ..schema.overlay import Box2D, OverlayMeta, OverlaySet

from .base import BaseAdapter


class CamelAdapter(BaseAdapter):
    split: str
    ann_path: str
    images_root: str

    def __init__(
        self,
        dataset_name: str = "camel",
        dataset_path: str = "data/camel_dataset",
        ann_file: str = "annotations/half-train_cocoformat.json",
        split: str = "train",
    ) -> None:
        self.split = split
        self.ann_path = f"{dataset_path}/{ann_file}"
        self.images_root = f"{dataset_path}/{split}"

        self.coco = COCO(str(self.ann_path))
        categories = sorted(self.coco.dataset.get("categories", []), key=lambda c: c["id"])
        self.category_id_to_class = {category["id"]: idx for idx, category in enumerate(categories)}

        super().__init__(dataset_name, dataset_path)

    def get_sequence_ids(self) -> list[str]:
        video_ids = {img.get("video_id") for img in self.coco.imgs.values()}
        return sorted([vid for vid in video_ids if vid is not None])

    def get_sensor_ids(self) -> list[str]:
        # Since we only have 1 camera
        return ["cam"]

    def index_frames(self) -> list[dict]:
        def sort_key(img: dict[str, Any]) -> tuple:
            """
            Simple function that returns a tuple of video & frame id
            """
            video_id = img.get("video_id", "")
            frame_id = img.get("frame_id")
            if frame_id is None:
                frame_id = img.get("mot_frame_id", img["id"])
            return (video_id, frame_id)

        frames = []
        images = cast(list[dict[str, Any]], list(self.coco.imgs.values()))
        for img in sorted(images, key=sort_key):
            # Get frame id & timestamp
            frame_id = img.get("frame_id")
            if frame_id is None:
                frame_id = img.get("mot_frame_id", img["id"])
            timestamp = float(img.get("mot_frame_id", frame_id))

            # Add the new frame
            frames.append(
                {
                    "image_id": img["id"],
                    "timestamp": timestamp,
                    "frame_id": int(frame_id),
                    "meta": FrameMeta(
                        sequence_id=img.get("video_id"),
                        dataset="camel",
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
        image_path = f"{self.images_root}/{img_info['file_name']}"
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Defining intrinsics & pose in ego
        intrinsics = np.eye(3, dtype=np.float32)
        sensor_pose_in_ego = np.eye(4, dtype=np.float32)

        meta = ImageMeta(spectral="rgb", frame="sensor:cam", intrinsics=intrinsics, sensor_pose_in_ego=sensor_pose_in_ego)
        data = ImageSensorFrame(sensor_id="cam", meta=meta, image=image, mask=None)
        return {"cam": SensorFrame(kind="image", data=data)}

    def load_overlays(self, info: dict) -> OverlaySet | None:
        ann_ids = self.coco.getAnnIds(imgIds=[info["image_id"]])
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        for ann in anns:
            # Skip the crowds
            if ann.get("iscrowd"):
                continue

            # Get the class id
            category_id = ann["category_id"]
            class_id = self.category_id_to_class.get(category_id)
            if class_id is None:
                continue

            # Converting ot xyxy format
            x, y, w, h = ann["bbox"]
            xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)

            meta = OverlayMeta(coord_frame="sensor:cam", source="gt", timestamp=info["timestamp"], sensor_id="cam")
            boxes.append(Box2D(meta=meta, xyxy=xyxy, class_id=class_id, confidence=None, track_id=ann.get("track_id")))

        return OverlaySet(boxes3D={}, boxes2D={"gt": boxes}, oriented_boxes2d={}, radar_dets={}, radar_polar_dets={}, tracks={})


# Testing the Adapter to see if everything looks good
# Can be ran from root using
#   python -m src.data.adapters.camel
if __name__ == "__main__":
    camel_adapter = CamelAdapter()

    # Checking the sensor ids
    print("--- Sensor Ids ---")
    sensor_ids = camel_adapter.get_sensor_ids()
    for sensor_id in sensor_ids:
        print(sensor_id)
    print("")

    # Checking the sequence ids
    print("--- Sequence Ids ---")
    sequence_ids = camel_adapter.get_sequence_ids()
    for sequence_id in sequence_ids:
        print(sequence_id)
    print("")

    # Looping through each frame
    print("--- Frames ---")
    first_frame = True
    line_counter = 0

    for frame in camel_adapter:  # using __getitem__ defined in BaseAdapter
        # ensuring that we only loop through the first video
        # we are doing this since frame id restarts for each video sequence
        if frame.frame_id == 0:
            if not first_frame:
                break
            first_frame = False

        print(frame.frame_id, end=" ")

        # Ensuring printing n number of ids per line
        line_counter += 1
        if line_counter == 20:
            print("")
            line_counter = 0
    print("")
