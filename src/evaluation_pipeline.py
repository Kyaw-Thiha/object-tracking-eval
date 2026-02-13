import sys
from pathlib import Path
import os

import torch
import numpy as np
import cv2 as cv
import time
import json
from scipy import stats as scipy_stats
from dataclasses import asdict

# tracker algorithms
from model.tracker.uncertainty_tracker import UncertaintyTracker
from model.tracker.prob_byte_tracker import ProbabilisticByteTracker
from model.tracker.prob_ocsort_tracker import ProbabilisticOCSORTTracker
from model.tracker.rcbevdet_3d_tracker import RCBEVDet3DTracker
from model.kalman_filter_uncertainty import KalmanFilterWithUncertainty

from core.utils.transforms import outs2results
from data.schema.prediction_3d import Detection3D, Track3D

import argparse
import importlib
import inspect

from evaluation_metrics.evaluate import evaluate

import faulthandler

faulthandler.enable()

allowed_trackers = [
    "none",
    "uncertainty_tracker",
    "probabilistic_byte_tracker",
    "prob_ocsort_tracker",
    "rcbevdet_3d_tracker",
]
allowed_2d_trackers = ["uncertainty_tracker", "probabilistic_byte_tracker", "prob_ocsort_tracker"]

# Ensure repo src is on sys.path when running from repo root or src/.
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def import_dataloader(factory_name: str):
    module_path = f"data.dataloaders.{factory_name}"

    try:
        module = importlib.import_module(module_path.replace(".py", ""))
        dataloader = module.factory()
        # TODO: checks to varify dataloader is indeed a torch dataloader
        return dataloader
    except ModuleNotFoundError:
        raise ValueError(f"Factory '{factory_name}' not found in data.dataloaders/")
    except AttributeError:
        raise ValueError(f"Factory '{factory_name}' does not define a 'factory' function")


def import_model_factory(factory_name: str):
    """
    Load a factory module from model/factory directory.

    Example: factory_name="opencv_yolox_factory_image_noise"
    """
    module_path = f"model.factory.{factory_name.replace('.py', '')}"

    try:
        module = importlib.import_module(module_path)
        return module.factory
    except ModuleNotFoundError:
        raise ValueError(f"Factory '{factory_name}' not found in model.factory/")
    except AttributeError:
        raise ValueError(f"Factory '{factory_name}' does not define a 'factory' function")


def load_tracker(tracker_name: str):
    if tracker_name == "none":
        return None
    if tracker_name == "rcbevdet_3d_tracker":
        print("Using RCBEVDet 3D Tracker")
        return RCBEVDet3DTracker()

    if tracker_name == "uncertainty_tracker":
        print("Using Uncertainty Tracker")
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

    elif tracker_name == "probabilistic_byte_tracker":
        print("Using Probabilistic Byte Tracker")
        tracker = ProbabilisticByteTracker(
            obj_score_thrs=dict(high=0.6, low=0.1),
            init_track_thr=0.7,
            weight_iou_with_det_scores=True,
            match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        )
        motion_model = KalmanFilterWithUncertainty(fps=30)
        tracker.motion = motion_model  # type: ignore[reportArgumentType]
    elif tracker_name == "prob_sort_tracker":
        print("Using Probabilistic Sort Tracker")
        raise ValueError("Probabilistic Sort Tracker is not yet implemented.")

    elif tracker_name == "prob_ocsort_tracker":
        print("Using Probabilistic OCSORT Tracker")
        tracker = ProbabilisticOCSORTTracker()
    else:
        raise ValueError(f"Unknown factory '{tracker_name}'. Choose one of: {allowed_trackers}")
    motion_model = KalmanFilterWithUncertainty(fps=30)
    tracker.motion = motion_model  # type: ignore[reportArgumentType]
    return tracker


class MOTDetector(torch.nn.Module):
    """
    A Wrapper class for the bare detection model to include motion model, this is the expected structure by the tracker.
    """

    def __init__(self, detector, motion):
        super().__init__()
        self.detector = detector
        self.motion = motion
        self.with_motion = True

    def forward(self, x, targets=None):
        # Route optional batch context to detectors that support it.
        # Mainly meant to be used by early/proposal-based fusion detectors
        if hasattr(self.detector, "infer_with_context"):
            preds = self.detector.infer_with_context(x, targets)
        else:
            preds = self.detector.infer(x)
        return preds


def parse_args():
    parser = argparse.ArgumentParser(description="evaluation pipeline")
    parser.add_argument("--dataloader_factory", required=True, help="dataloader factory file name under data/dataloaders directory.", type=str)
    parser.add_argument("--dataset_dir", required=True, help="directory path where the dataset is stored.", type=str)
    parser.add_argument("--model_factory", required=True, help="model factory file name under model/factory directory.", type=str)
    parser.add_argument("--tracker", default="none", help="specify a tracker type", choices=allowed_trackers, type=str)
    parser.add_argument("--output_mode", default="2d", choices=["2d", "3d"], type=str)
    parser.add_argument("--device", required=True, help="specific device type", choices=["cpu", "cuda"], type=str)
    parser.add_argument("--output_dir", required=True, help="directory path where tracking output will be stored", type=str)
    parser.add_argument("--eval_result_dir", required=True, help="directory path where to save the evaluation result to.", type=str)
    parser.add_argument("--plot_save_path", required=True, help="path to save the evaluation plots.", type=str)
    parser.add_argument("--checkpoint_path", default=None, help="optional model checkpoint path passed to model factory", type=str)
    parser.add_argument("--config_path", default=None, help="optional model config path passed to model factory", type=str)
    parser.add_argument(
        "--smoke_eval",
        action="store_true",
        help="allow smoke runs with uninitialized model weights when supported by model factory",
    )
    parser.add_argument("--bev_pool_backend", default="auto", choices=["auto", "cuda_ext", "torch"], type=str)
    parser.add_argument("--annotate-videos", action="store_true", help="Generate annotated videos with tracking results")
    parser.add_argument("--video-output-dir", default="./annotated_videos", help="Directory to save annotated videos", type=str)
    return parser.parse_args()


def is_directory_empty(path):
    """Checks if a given directory is empty."""
    if not os.path.isdir(path):
        return True
    with os.scandir(path) as entries:
        return not any(entries)


def _tensor_to_list(t: torch.Tensor):
    return t.detach().cpu().tolist()


def _serialize_detection_3d(frame_id: int, video_id: str, det: dict) -> dict:
    obj = Detection3D(
        frame_id=int(frame_id),
        video_id=str(video_id),
        boxes_3d=det["boxes_3d"].detach().cpu(),
        scores_3d=det["scores_3d"].detach().cpu(),
        labels_3d=det["labels_3d"].detach().cpu(),
        velocities=det.get("velocities").detach().cpu() if det.get("velocities") is not None else None,
    )
    payload = asdict(obj)
    payload["boxes_3d"] = _tensor_to_list(obj.boxes_3d)
    payload["scores_3d"] = _tensor_to_list(obj.scores_3d)
    payload["labels_3d"] = _tensor_to_list(obj.labels_3d)
    if obj.velocities is not None:
        payload["velocities"] = _tensor_to_list(obj.velocities)
    return payload


def _serialize_track_3d(frame_id: int, video_id: str, trk: dict) -> dict:
    obj = Track3D(
        frame_id=int(frame_id),
        video_id=str(video_id),
        boxes_3d=trk["boxes_3d"].detach().cpu(),
        scores_3d=trk["scores_3d"].detach().cpu(),
        labels_3d=trk["labels_3d"].detach().cpu(),
        track_ids=trk["track_ids"].detach().cpu(),
        velocities=trk.get("velocities").detach().cpu() if trk.get("velocities") is not None else None,
    )
    payload = asdict(obj)
    payload["boxes_3d"] = _tensor_to_list(obj.boxes_3d)
    payload["scores_3d"] = _tensor_to_list(obj.scores_3d)
    payload["labels_3d"] = _tensor_to_list(obj.labels_3d)
    payload["track_ids"] = _tensor_to_list(obj.track_ids)
    if obj.velocities is not None:
        payload["velocities"] = _tensor_to_list(obj.velocities)
    return payload


def main():
    args = parse_args()

    device = args.device
    print("[INFO] Using device:", device)

    if args.output_mode == "2d" and args.tracker not in allowed_2d_trackers:
        raise ValueError(f"2D mode requires one of {allowed_2d_trackers}. Got '{args.tracker}'.")
    if args.output_mode == "3d" and args.tracker in allowed_2d_trackers:
        raise ValueError("3D mode supports '--tracker none' or '--tracker rcbevdet_3d_tracker' only.")

    results = {}
    output_dir = args.output_dir
    result_path = os.path.join(args.eval_result_dir, "evaluation_result.json")

    if args.output_mode == "2d" and not is_directory_empty(output_dir) and os.path.exists(result_path):
        print(f"[INFO] Output directory {output_dir} is not empty. Skipping inference and tracking.")
        with open(result_path, "r") as f:
            results = json.load(f)
    else:
        print(f"[INFO] Starting {args.output_mode.upper()} evaluation pipeline...")

        # --- Get model from factory ---
        factory = import_model_factory(args.model_factory)
        factory_sig = inspect.signature(factory)
        factory_kwargs = {"device": device}
        if "checkpoint_path" in factory_sig.parameters and args.checkpoint_path is not None:
            factory_kwargs["checkpoint_path"] = args.checkpoint_path
        if "config_path" in factory_sig.parameters and args.config_path is not None:
            factory_kwargs["config_path"] = args.config_path
        if "smoke_eval" in factory_sig.parameters:
            factory_kwargs["smoke_eval"] = args.smoke_eval
        if "bev_pool_backend" in factory_sig.parameters:
            factory_kwargs["bev_pool_backend"] = args.bev_pool_backend

        model = factory(**factory_kwargs)
        assert hasattr(model, "get_classes"), "ASSERT ERROR: The model class must have method: get_classes() -> list[str]"
        if args.output_mode == "2d":
            assert hasattr(model, "infer"), "ASSERT ERROR: 2D model must have infer()"
        else:
            assert hasattr(model, "infer_with_context_3d"), "ASSERT ERROR: 3D model must have infer_with_context_3d()"

        class_names = model.get_classes()
        number_classes = len(class_names)

        # --- Initialize tracker ---
        tracker = load_tracker(args.tracker)
        # Wrapper on detector to include motion model, expected by 2D trackers.
        mot_model = MOTDetector(detector=model, motion=KalmanFilterWithUncertainty()) if args.output_mode == "2d" else None
        if tracker is not None:
            tracker.reset()

        # --- Validate dataset path ---
        if not os.path.exists(args.dataset_dir):
            raise ValueError(f"Dataset directory {args.dataset_dir} does not exist.")

        # --- Build dataloader ---
        dataloader = import_dataloader(args.dataloader_factory)

        os.makedirs(output_dir, exist_ok=True)
        start_time = time.time()
        if args.output_mode == "2d":
            # 2D MOT path (existing behavior)
            current_video = None
            results_per_video = []
            cur_video_frames = 0
            throughputs = []
            video_start = start_time

            # --- Inference + tracking ---
            with torch.no_grad():
                for i, (imgs, targets) in enumerate(dataloader):
                    imgs = imgs.to(device)
                    actual_batch_size = imgs.shape[0]
                    cur_video_frames += imgs.shape[0]

                    # Detector output contract: (bboxes, labels, covariances)
                    batch_dets = mot_model(imgs, targets)  # type: ignore[misc]
                    assert isinstance(batch_dets, tuple) and len(batch_dets) == 3
                    batch_bboxes, batch_labels, batch_covs = batch_dets
                    assert len(batch_bboxes) == len(batch_labels) == len(batch_covs) == actual_batch_size

                    # --- Iterate frames in current batch ---
                    for j in range(actual_batch_size):
                        img = imgs[j]
                        det_bboxes = batch_bboxes[j]
                        det_labels = batch_labels[j]
                        bbox_covs = batch_covs[j]
                        img_meta = targets[j]["img_metas"]
                        frame_id = targets[j]["frame_id"]
                        frame_id_tracker = frame_id - 1
                        video_id = targets[j]["video_id"]
                        scale_factor = img_meta["scale_factor"]

                        if len(det_bboxes) == 0:
                            continue
                        # Scale detector bboxes back to original image space.
                        det_bboxes = det_bboxes.clone()
                        det_bboxes[:, :4] /= det_bboxes.new_tensor(scale_factor)

                        # --- Tracking ---
                        track_bboxes, track_bbox_covs, track_labels, track_ids = tracker.track(
                            img,
                            img_metas=[img_meta],
                            model=mot_model,
                            bboxes=det_bboxes,
                            bbox_covs=bbox_covs,
                            labels=det_labels,
                            frame_id=frame_id_tracker,
                            rescale=False,
                        )

                        track_results = outs2results(
                            bboxes=track_bboxes,
                            bbox_covs=track_bbox_covs,
                            labels=track_labels,
                            ids=track_ids,
                            num_classes=number_classes,
                        )

                        # Reset tracker and flush file when sequence changes.
                        if current_video != video_id:
                            if current_video is not None:
                                out_path = os.path.join(output_dir, f"{current_video}.txt")
                                with open(out_path, "w") as f:
                                    for row in results_per_video:
                                        f.write(",".join(map(str, row)) + "\n")
                                print(f"[INFO] Saved results for video {current_video} to {out_path}")
                                results_per_video = []
                                tracker.reset()
                                video_end = time.time()
                                elapsed = video_end - video_start
                                throughputs.append(cur_video_frames / elapsed if elapsed > 0 else 0.0)
                                video_start = time.time()
                                cur_video_frames = 0

                            current_video = video_id
                            print(f"[INFO] Starting new sequence {video_id}")

                        # Append MOT-format rows for this frame.
                        for class_bboxes in track_results["bbox_results"]:
                            for row in class_bboxes:
                                tid = int(row[0])
                                x1, y1, x2, y2, score = row[1:]
                                w, h = x2 - x1, y2 - y1
                                results_per_video.append(
                                    [int(frame_id), tid, float(x1), float(y1), float(w), float(h), float(score), 1, -1]
                                )

                    if i % 10 == 0:
                        print(f"Processed {i} batches")

                # --- Flush last active sequence ---
                if current_video is not None and results_per_video:
                    out_path = os.path.join(output_dir, f"{current_video}.txt")
                    with open(out_path, "w") as f:
                        for row in results_per_video:
                            f.write(",".join(map(str, row)) + "\n")
                    print(f"[INFO] Saved results for video {current_video} to {out_path}")

                end_time = time.time()
                print(f"[INFO] Inference and tracking completed in {end_time - start_time:.2f} seconds.")

            if len(throughputs) >= 2:
                throughput_ci = scipy_stats.bootstrap((throughputs,), np.mean, confidence_level=0.95)
                results["throughput_ci"] = {
                    "low": float(throughput_ci.confidence_interval.low),
                    "high": float(throughput_ci.confidence_interval.high),
                }
                results["throughput (frame per second)"] = {
                    "conference interval": [throughput_ci.confidence_interval.low, throughput_ci.confidence_interval.high]
                }
            else:
                print("[WARN] Not enough sequences to compute throughput CI; skipping bootstrap.")
            results["inference time (seonds)"] = end_time - start_time
        else:
            # 3D-native path (detector-only or detector+3D tracker)
            current_video = None
            lines_per_video: list[str] = []
            num_frames = 0
            num_dets = 0
            num_tracks = 0

            # --- 3D inference ---
            with torch.no_grad():
                for i, (imgs, targets) in enumerate(dataloader):
                    imgs = imgs.to(device)
                    batch_out = model.infer_with_context_3d(imgs, targets)
                    assert isinstance(batch_out, list) and len(batch_out) == imgs.shape[0]

                    # --- Iterate frames in current batch ---
                    for j, frame_out in enumerate(batch_out):
                        frame_id = int(targets[j]["frame_id"])
                        video_id = str(targets[j]["video_id"])
                        num_frames += 1
                        num_dets += int(frame_out["boxes_3d"].shape[0])

                        # Reset 3D tracker and flush file when sequence changes.
                        if current_video != video_id:
                            if current_video is not None:
                                out_path = os.path.join(output_dir, f"{current_video}.jsonl")
                                with open(out_path, "w") as f:
                                    for line in lines_per_video:
                                        f.write(line + "\n")
                                lines_per_video = []
                                if tracker is not None:
                                    tracker.reset()
                            current_video = video_id
                            print(f"[INFO] Starting new sequence {video_id}")

                        # Save per-frame 3D detections.
                        det_payload = {"type": "det3d", **_serialize_detection_3d(frame_id=frame_id, video_id=video_id, det=frame_out)}
                        lines_per_video.append(json.dumps(det_payload))

                        # Optional per-frame 3D tracking result.
                        if tracker is not None:
                            trk_out = tracker.track(frame_out, frame_id=frame_id)
                            num_tracks += int(trk_out["boxes_3d"].shape[0])
                            trk_payload = {"type": "track3d", **_serialize_track_3d(frame_id=frame_id, video_id=video_id, trk=trk_out)}
                            lines_per_video.append(json.dumps(trk_payload))

                    if i % 10 == 0:
                        print(f"Processed {i} batches")

                # --- Flush last active sequence ---
                if current_video is not None and lines_per_video:
                    out_path = os.path.join(output_dir, f"{current_video}.jsonl")
                    with open(out_path, "w") as f:
                        for line in lines_per_video:
                            f.write(line + "\n")
                    print(f"[INFO] Saved 3D results for video {current_video} to {out_path}")

                end_time = time.time()
                results["inference_time_seconds"] = end_time - start_time
                results["num_frames"] = num_frames
                results["num_detections_3d"] = num_dets
                results["num_tracks_3d"] = num_tracks
                print(f"[INFO] 3D inference completed in {end_time - start_time:.2f} seconds.")

    # --- Evaluation and metrics ---
    if args.output_mode == "2d":
        results.update(evaluate(output_dir, args.dataset_dir))
    else:
        results["note"] = "3D outputs saved as JSONL per sequence. No MOT17 2D evaluator run in 3D mode."

    os.makedirs(args.eval_result_dir, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(results, f)

    # --- Optionally generate annotated videos ---
    if args.annotate_videos and args.output_mode == "2d":
        print("[INFO] Generating annotated videos...")
        # Get sequence names from output directory
        seq_names = [f.stem for f in Path(output_dir).glob("*.txt")]
        annotate_results_from_txt(dataset_root=args.dataset_dir, resfile_dir=output_dir, seq_names=seq_names, output_dir=args.video_output_dir)


def annotate_results_from_txt(dataset_root, resfile_dir, seq_names, output_dir):
    """Annotate videos with tracking results.

    Args:
        dataset_root: Path to dataset root (e.g., MOT17/train)
        resfile_dir: Directory containing .txt result files
        seq_names: List of sequence names (e.g., ['MOT17-02-SDP', ...])
        output_dir: Directory to save annotated videos
    """
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)

    for seq_name in seq_names:
        gt_file = os.path.join(dataset_root, seq_name, "gt", "gt.txt")
        img_dir = os.path.join(dataset_root, seq_name, "img1")
        track_file = os.path.join(resfile_dir, f"{seq_name}.txt")

        if not os.path.exists(track_file):
            print(f"[SKIP] {seq_name}: tracking file not found")
            continue

        if not os.path.exists(img_dir):
            print(f"[SKIP] {seq_name}: image directory not found")
            continue

        # Load GT (if available)
        gt_dict = {}
        if os.path.exists(gt_file):
            with open(gt_file) as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 6:
                        continue
                    frame_id = int(float(parts[0]))
                    x, y, w, h = map(float, parts[2:6])
                    gt_dict.setdefault(frame_id, []).append([x, y, w, h])

        # Load tracking results
        track_dict = {}
        with open(track_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                frame_id = int(float(parts[0]))
                track_id = int(float(parts[1]))
                x, y, w, h = map(float, parts[2:6])
                track_dict.setdefault(frame_id, []).append([x, y, w, h, track_id])

        # Get image files
        image_files = sorted([f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))])
        if not image_files:
            print(f"[SKIP] {seq_name}: no images found")
            continue

        num_frames = len(image_files)
        sample_img = cv.imread(os.path.join(img_dir, image_files[0]))
        H, W = sample_img.shape[:2]

        # Setup video writer
        out_path = os.path.join(output_dir, f"{seq_name}_annotated.mp4")
        writer = cv.VideoWriter(out_path, getattr(cv, "VideoWriter_fourcc")(*"mp4v"), 30, (W, H))

        for frame_id in tqdm(range(1, num_frames + 1), desc=f"[{seq_name}]"):
            img_path = os.path.join(img_dir, f"{frame_id:06d}.jpg")
            if not os.path.exists(img_path):
                # Try .png extension
                img_path = os.path.join(img_dir, f"{frame_id:06d}.png")
                if not os.path.exists(img_path):
                    continue

            img = cv.imread(img_path)

            # Draw GT (green)
            for x, y, w, h in gt_dict.get(frame_id, []):
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(img, "GT", (x1, y1 - 2), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Draw Tracks (red)
            for x, y, w, h, track_id in track_dict.get(frame_id, []):
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv.putText(img, f"ID {track_id}", (x1, y2 + 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            writer.write(img)

        writer.release()
        print(f"[✓] Saved: {out_path}")


if __name__ == "__main__":
    main()
