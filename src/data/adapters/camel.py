from typing import Any, cast
from pathlib import Path
from .coco import CocoBaseAdapter


class CamelAdapter(CocoBaseAdapter):
    def __init__(
        self,
        dataset_name: str = "camel",
        dataset_path: str = "data/camel_dataset",
        ann_file: str = "annotations/half-train_cocoformat.json",
        split: str = "train",
    ) -> None:
        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            ann_file=ann_file,
            split=split,
            images_root=f"{dataset_path}/{split}",
        )

    # Optional: be strict about using mot_frame_id for time if you want
    def get_timestamp(self, img: dict[str, Any]) -> float:
        mot_frame_id = img.get("mot_frame_id")
        if mot_frame_id is None:
            return float(self.get_frame_id(img))
        return float(mot_frame_id)

    def get_frame_id(self, img: Any) -> int:
        img = cast(dict[str, Any], img)
        mot_frame_id = img.get("mot_frame_id")
        if mot_frame_id is not None:
            return int(mot_frame_id)
        return super().get_frame_id(img)

    def get_image_path(self, img: Any):
        img = cast(dict[str, Any], img)
        file_name = img["file_name"]
        if file_name.startswith(("train/", "test/")):
            return Path(self.dataset_path) / file_name
        return Path(self.dataset_path) / self.split / file_name


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
