import sys
from pathlib import Path
import os

import torch
import numpy as np
from PIL import Image
import cv2 as cv
import time
import json
import scipy
from scipy import stats as scipy_stats

# tracker algorithms
from model.tracker.uncertainty_tracker import UncertaintyTracker
from model.tracker.prob_byte_tracker import ProbabilisticByteTracker
from model.tracker.prob_sort_tracker import ProbabilisticSortTracker
from model.tracker.prob_ocsort_tracker import ProbabilisticOCSORTTracker
from model.kalman_filter_uncertainty import KalmanFilterWithUncertainty

from core.utils import results2outs, outs2results

from data.datasets.mot17_dataset import MOT17CocoDataset
from torch.utils.data import DataLoader

import argparse
import importlib

from evaluation_metrics.evaluate import evaluate

import faulthandler

faulthandler.enable()

allowed_trackers = ["uncertainty_tracker", "probabilistic_byte_tracker", "prob_ocsort_tracker"]

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
    Load a factory module from model_factory directory.

    Example: factory_name="opencv_yolox_factory_image_noise"
    """
    module_path = f"model_factory.{factory_name.replace('.py', '')}"

    try:
        module = importlib.import_module(module_path)
        return module.factory
    except ModuleNotFoundError:
        raise ValueError(f"Factory '{factory_name}' not found in model_factory/")
    except AttributeError:
        raise ValueError(f"Factory '{factory_name}' does not define a 'factory' function")


def load_tracker(tracker_name: str):
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
        raise ValueError(f"Probabilistic Sort Tracker is not yet implemented.")

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

    def forward(self, x):
        # inference
        preds = self.detector.infer(x)
        return preds


def parse_args():
    parser = argparse.ArgumentParser(description="evaluation pipeline")
    parser.add_argument(
        "--dataloader_factory", required=True, help="dataloader factory file name under data/dataloaders directory.", type=str
    )
    parser.add_argument("--dataset_dir", required=True, help="directory path where the dataset is stored.", type=str)
    parser.add_argument("--model_factory", required=True, help="modle factory file name under model_factory directory.", type=str)
    parser.add_argument("--tracker", required=True, help="specify a tracker type", choices=allowed_trackers, type=str)
    parser.add_argument("--device", required=True, help="specific device type", choices=["cpu", "cuda"], type=str)
    parser.add_argument("--output_dir", required=True, help="directory path where tracking output will be stored", type=str)
    parser.add_argument(
        "--eval_result_dir", required=True, help="directory path where to save the evaluation result to.", type=str
    )
    parser.add_argument("--plot_save_path", required=True, help="path to save the evaluation plots.", type=str)
    return parser.parse_args()


def is_directory_empty(path):
    """Checks if a given directory is empty."""
    if not os.path.isdir(path):
        return True
    with os.scandir(path) as entries:
        return not any(entries)


def main(debug=False):
    args = parse_args()

    device = args.device
    print("[INFO] Using device:", device)

    results = {}
    output_dir = args.output_dir
    result_path = os.path.join(args.eval_result_dir, "evaluation_result.json")

    if not is_directory_empty(output_dir) and os.path.exists(result_path):
        print(f"[INFO] Output directory {output_dir} is not empty. Skipping inference and tracking.")
        with open(result_path, "r") as f:
            results = json.load(f)

    else:
        print(f"[INFO] Starting tracking evaluation pipeline...")
        # --- Get model from factory ---
        # from model_factory.opencv_yolox_factory import factory
        factory = import_model_factory(args.model_factory)

        model = factory(device=device)
        assert hasattr(model, "get_classes"), "ASSERT ERROR: The model class must have method: get_classes() -> list[str]"
        assert hasattr(model, "infer"), "ASSERT ERROR: The model class must have method: infer() -> tuple[list]"

        class_names = model.get_classes()
        number_classes = len(class_names)

        # --- Initialize Tracker
        tracker = load_tracker(args.tracker)

        # Wrapper on detector to include motion model, expected by the tracker
        mot_model = MOTDetector(detector=model, motion=KalmanFilterWithUncertainty())

        tracker.reset()

        # check dataset dir
        if not os.path.exists(args.dataset_dir):
            raise ValueError(f"Dataset directory {args.dataset_dir} does not exist.")

        dataloader = import_dataloader(args.dataloader_factory)
        batch_size = dataloader.batch_sampler.batch_size

        print("[DEBUG] Checking dataloader...")
        print("  Batch sampler batch_size:", dataloader.batch_sampler.batch_size)
        print("  Dataset length:", len(dataloader.dataset))
        # Try to fetch first batch
        try:
            first_batch = next(iter(dataloader))
            print("[DEBUG] Successfully loaded first batch")
            print("  imgs shape:", first_batch[0].shape)
            print("  number of targets:", len(first_batch[1]))
        except StopIteration:
            print("[ERROR] Dataloader is EMPTY! No samples found.")
        except Exception as e:
            print("[ERROR] Exception while loading first batch:", e)

        current_video = None  # flag to indicate when a new video sequence starts

        results_per_video = []  # saves tracking results for the current video

        os.makedirs(output_dir, exist_ok=True)

        start_time = time.time()
        total_frames_processed = 0
        cur_video_frames = 0
        cur_video_run_time = 0
        throughputs = []
        video_start = start_time
        # --- Inference ---
        with torch.no_grad():
            for i, (imgs, targets) in enumerate(dataloader):
                imgs = imgs.to(device)  # (B, 3, H, W)

                total_frames_processed += imgs.shape[0]
                cur_video_frames += imgs.shape[0]

                actual_batch_size = imgs.shape[0]

                # --- Inference ---
                batch_dets = mot_model(imgs)

                assert isinstance(batch_dets, tuple) and len(batch_dets) == 3, (
                    "Model Inference must return a tuple of (batch_detection_bboxes, batch_detection_labels, batch_detection_bbox_covariance_matrices)"
                )

                batch_bboxes, batch_labels, batch_covs = batch_dets

                assert isinstance(batch_bboxes, list), "batch bboxes must be a python list"
                assert all(isinstance(t, torch.Tensor) for t in batch_bboxes), "batch_bboxes must contain only torch.Tensor"
                assert isinstance(batch_labels, list), "batch labels must be a python list"
                assert all(isinstance(t, torch.Tensor) for t in batch_labels), "batch_labels must contain only torch.Tensor"
                assert isinstance(batch_covs, list), "batch covs must be a python list"
                assert all(isinstance(t, torch.Tensor) for t in batch_covs), "batch_covs must contain only torch.Tensor"

                if not (
                    len(batch_bboxes) == len(batch_labels)
                    and len(batch_labels) == len(batch_covs)
                    and len(batch_covs) == actual_batch_size
                ):
                    print(f"len(batch_bboxes): {len(batch_bboxes)}")
                    print(f"len(batch_labels): {len(batch_labels)}")
                    print(f"len(batch_covs): {len(batch_covs)}")
                    print(f"batch_size: {batch_size}")
                    print(f"actual_batch_size: {actual_batch_size}")

                assert (
                    len(batch_bboxes) == len(batch_labels)
                    and len(batch_labels) == len(batch_covs)
                    and len(batch_covs) == actual_batch_size
                ), "result lists lengths must be identical"

                # --- iterate over frames within the batch  ---
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

                    # scale bboxes back
                    if len(det_bboxes) == 0:
                        continue
                    scaled_bboxes = det_bboxes.clone()
                    # print("Before unletterbox:", scaled_bboxes.shape)
                    scaled_bboxes[:, :4] /= det_bboxes.new_tensor(scale_factor)
                    det_bboxes = scaled_bboxes

                    # print(f"Tracking starts, video_id: {video_id}, frame_id: {frame_id}, num_detections: {det_bboxes.shape[0]}")

                    # --- tracking ---
                    track_bboxes, track_bbox_covs, track_labels, track_ids = tracker.track(
                        img=img,
                        img_metas=[img_meta],
                        model=mot_model,
                        bboxes=det_bboxes,
                        bbox_covs=bbox_covs,
                        labels=det_labels,
                        frame_id=frame_id_tracker,
                        rescale=False,
                    )
                    # print(f"After tracking, num_tracks: {track_bboxes.shape[0]}")

                    track_results = outs2results(
                        bboxes=track_bboxes,
                        bbox_covs=track_bbox_covs,
                        labels=track_labels,
                        ids=track_ids,
                        num_classes=number_classes,
                    )

                    # reset tracker when encoutering frames from a new video
                    if current_video != video_id:
                        # write previous video results to file
                        if current_video is not None:
                            out_path = os.path.join(output_dir, f"{current_video}.txt")
                            with open(out_path, "w") as f:
                                for row in results_per_video:
                                    f.write(",".join(map(str, row)) + "\n")
                            print(f"[INFO] Saved results for video {current_video} to {out_path}")
                            results_per_video = []

                            # reset tracker for new video
                            tracker.reset()
                            video_end = time.time()
                            cur_video_run_time = video_end - video_start
                            throughputs.append(cur_video_frames / cur_video_run_time)
                            video_start = time.time()
                            cur_video_frames = 0

                        current_video = video_id
                        print(f"[INFO] Starting new sequence {video_id}")

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
                print(f"[INFO] Saved results for video {current_video} to {out_path}")

            # end inference timing
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

    # --- After all videos are processed, perform evaluation ---
    results.update(evaluate(output_dir, args.dataset_dir))
    os.makedirs(args.eval_result_dir, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main(debug=False)
