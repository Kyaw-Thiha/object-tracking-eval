"""
This is a test script in the development of a simplified inference pipeline
"""

import os
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import cv2 as cv

from model.tracker.uncertainty_tracker import UncertaintyTracker
from model.kalman_filter_uncertainty import KalmanFilterWithUncertainty

from data.datasets.mot17_dataset import MOT17CocoDataset
from torch.utils.data import DataLoader
from yolox import YoloX

SRC_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_ROOT.parent
REPO_ROOT = SRC_ROOT.parent
YOLOX_ROOT = REPO_ROOT / "object_detection_yolox"
sys.path.append(str(YOLOX_ROOT))

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("Using device:", device)

# Check OpenCV version
opencv_python_version = lambda str_version: tuple(map(int, (str_version.split("."))))
assert opencv_python_version(cv.__version__) >= opencv_python_version("4.10.0"), (
    "Please install latest opencv-python for benchmark: python3 -m pip install --upgrade opencv-python"
)

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX, cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN, cv.dnn.DNN_TARGET_NPU],
]


def letterbox(srcimg, target_size=(640, 640)):
    padded_img = np.ones((target_size[0], target_size[1], 3)).astype(np.float32) * 114.0
    ratio = min(target_size[0] / srcimg.shape[0], target_size[1] / srcimg.shape[1])
    resized_img = cv.resize(
        srcimg, (int(srcimg.shape[1] * ratio), int(srcimg.shape[0] * ratio)), interpolation=cv.INTER_LINEAR
    ).astype(np.float32)
    padded_img[: int(srcimg.shape[0] * ratio), : int(srcimg.shape[1] * ratio)] = resized_img

    return padded_img, ratio


def unletterbox(bbox, letterbox_scale):
    return bbox / letterbox_scale


backend_target = 0

backend_id = backend_target_pairs[backend_target][0]
target_id = backend_target_pairs[backend_target][1]


class MOTDetector(torch.nn.Module):
    """
    A Wrapper class for the bare detection model to include motion model, this is the expected structure by the tracker.
    """

    def __init__(self, detector, motion):
        super().__init__()
        self.detector = detector
        self.motion = motion
        self.with_motion = True

    def forward(self, x):
        # inference
        preds = self.detector.infer(x)
        return preds


def build_mot17_dataloader(ann_file, img_prefix, batch_size=4, num_workers=4, input_size=(640, 640)):
    dataset = MOT17CocoDataset(ann_file, img_prefix, input_size=input_size)

    def detection_collate_fn(batch):
        images = torch.stack([b[0] for b in batch])  # stack image tensors
        targets = [b[1] for b in batch]  # keep list of dicts/arrays
        return images, targets

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=detection_collate_fn
    )
    return dataset, dataloader


def main(debug=False):
    mot17_root = PROJECT_ROOT / "data" / "MOT17"
    test_image_path = mot17_root / "test" / "MOT17-03-FRCNN" / "img1" / "000001.jpg"
    test_image = np.array(Image.open(test_image_path).convert("RGB"))
    model_checkpoint_path = YOLOX_ROOT / "object_detection_yolox_2022nov.onnx"

    # --- Build MOT17 dataset + dataloader ---
    ann_file_path = mot17_root / "annotations" / "half-train_cocoformat.json"
    image_prefix_path = mot17_root / "train"

    # --- Initialize modle for inference ---
    class_names = (
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    )

    number_classes = len(class_names)
    model = YoloX(
        modelPath=str(model_checkpoint_path),
        confThreshold=0.5,
        nmsThreshold=0.5,
        objThreshold=0.5,
        backendId=backend_id,
        targetId=target_id,
    )

    # --- Initialize Tracker
    tracker = UncertaintyTracker(
        obj_score_thr=0.3,
        init_track_thr=0.7,
        weight_iou_with_det_scores=True,
        match_iou_thr=0.3,
        num_tentatives=3,
        vel_consist_weight=0.2,
        vel_delta_t=3,
        num_frames_retain=30,
        with_covariance=True,
        det_score_mode="confidence",
        use_giou=False,
        expand_boxes=True,
        percent=0.3,
        ellipse_filter=True,
        filter_output=True,
        combine_mahalanobis=False,
        primary_cascade={"num_bins": None},  # dict, not False
        secondary_fn=None,
        secondary_cascade={"num_bins": None},  # dict, not False
    )
    motion_model = KalmanFilterWithUncertainty(fps=30)
    tracker.motion = motion_model

    # Wrapper on detector to include motion model, expected by the tracker
    mot_model = MOTDetector(detector=model, motion=KalmanFilterWithUncertainty())

    fixed_cov_coeff = 1.0
    batch_size = 4
    tracker.reset()

    dataset, dataloader = build_mot17_dataloader(
        str(ann_file_path),
        str(image_prefix_path),
        batch_size=batch_size,
    )

    current_video = None  # flag to indicate when a new video sequence starts

    results_per_video = []  # saves tracking results for the current video
    output_dir = "./outputs/simplyfied_pipeline_results"
    os.makedirs(output_dir, exist_ok=True)

    # --- Inference ---
    with torch.no_grad():
        for i, (imgs, targets) in enumerate(dataloader):
            imgs = imgs.to(device)  # (B=1, 3, H, W)

            # --- iterate over frames within the batch ---
            for j in range(batch_size):
                img = imgs[j]
                img_meta = targets[j]["img_metas"]
                frame_id = targets[j]["frame_id"]
                video_id = targets[j]["video_id"]
                scale_factor = img_meta["scale_factor"]

                # --- Inference ---
                img_np = img.detach().cpu().numpy().transpose(1, 2, 0)  # from torch (B, 3, H, W)[i] -> numpy (H, W, 3)
                dets = mot_model(img_np)

                det_bboxes = []
                det_labels = []

                for det in dets:
                    box = unletterbox(det[:4], scale_factor).astype(np.int32)
                    score = det[-2]
                    cls_id = int(det[-1])
                    x0, y0, x1, y1 = box
                    if cls_id == 0:
                        det_bboxes.append([x0, y0, x1, y1, score])
                        det_labels.append(cls_id)

                # skip empty detections
                if dets is None or dets.shape[0] == 0:
                    continue

                # prepare tensors for tracker
                det_bboxes = torch.tensor(det_bboxes, dtype=torch.float32, device=img.device)
                det_labels = torch.tensor(det_labels, dtype=torch.long, device=img.device)
                bbox_covs = torch.eye(4, device=img.device).unsqueeze(0).repeat(det_bboxes.shape[0], 1, 1)

                if debug:
                    print(f"Frame {frame_id} - Det bboxes shape: {det_bboxes.shape}, Labels shape: {det_labels.shape}")
                    print(f"Det bboxes shape: {det_bboxes.shape}, Labels shape: {det_labels.shape}")
                    print(f"bbox_covs shape: {bbox_covs.shape}")

                    print(
                        f"Det Bboxes top left: max=({min([bbox[0] for bbox in det_bboxes])}, {min([bbox[1] for bbox in det_bboxes])}"
                    )
                    print(
                        f"Det Bboxes bottom right: max=({max([bbox[2] for bbox in det_bboxes])}, {max([bbox[3] for bbox in det_bboxes])}"
                    )

                    print(f"Calling tracker.track for frame {frame_id}")
                    print(f"  det_bboxes shape: {det_bboxes.shape}")
                    print(f"  bbox_covs type: {type(bbox_covs)}, shape: {getattr(bbox_covs, 'shape', None)}")

                # --- tracking ---
                track_bboxes, track_bbox_covs, track_labels, track_ids = tracker.track(
                    img=img,
                    img_metas=[img_meta],
                    model=mot_model,
                    bboxes=det_bboxes,
                    bbox_covs=bbox_covs,
                    labels=det_labels,
                    frame_id=frame_id,
                    rescale=False,
                )

                from core.utils import results2outs, outs2results

                track_results = outs2results(
                    bboxes=track_bboxes, bbox_covs=track_bbox_covs, labels=track_labels, ids=track_ids, num_classes=number_classes
                )
                det_results = outs2results(bboxes=det_bboxes, labels=det_labels, num_classes=number_classes)

                # reset tracker when encoutering frames from a new video
                if current_video != video_id:
                    # write previous video results to file
                    if current_video is not None:
                        out_path = os.path.join(output_dir, f"{current_video}.txt")
                        with open(out_path, "w") as f:
                            for row in results_per_video:
                                f.write(",".join(map(str, row)) + "\n")
                        print(f"Saved results for video {current_video} to {out_path}")
                        results_per_video = []

                        # reset tracker for new video
                        tracker.reset()
                    current_video = video_id
                    print(f"Starting new sequence {video_id}")

                # append track_results for the current frame to all retults of the current video
                for class_bboxes in track_results["bbox_results"]:
                    for row in class_bboxes:
                        tid = int(row[0])  # track ID
                        x1, y1, x2, y2, score = row[1:]
                        w, h = x2 - x1, y2 - y1

                        mot_row = [
                            int(frame_id),
                            tid,
                            float(x1),
                            float(y1),
                            float(w),
                            float(h),
                            float(score),
                            1,  # class id (pedestrian for MOT17)
                            -1,  # visibility (not provided)
                        ]
                        results_per_video.append(mot_row)

            if i % 10 == 0:
                print(f"Processed {i} batches")

        # --- After the entire evaluation loop finishes ---
        if current_video is not None and results_per_video:
            out_path = os.path.join(output_dir, f"{current_video}.txt")
            with open(out_path, "w") as f:
                for row in results_per_video:
                    f.write(",".join(map(str, row)) + "\n")
            print(f"Saved results for video {current_video} to {out_path}")


if __name__ == "__main__":
    main(debug=False)
