"""
This script uses HOTA from TrackEval to compute HOTA and AssA metrics for multi-object tracking results.
"""


from pathlib import Path
import numpy as np
import scipy.stats

from evaluation_metrics.utils import (
    read_annotations,
    get_gt_files_from_dir,
    get_output_files_from_dir,
    compute_iou_matrix,
)

from trackeval.metrics.hota import HOTA

SRC_ROOT = Path(__file__).resolve().parents[1]


def build_hota_data(gt_ann, pred_ann):

    all_frame_ids = sorted(set(gt_ann.keys()) | set(pred_ann.keys()))

    gt_ids = []
    tracker_ids = []
    similarity_scores = []

    gt_global_ids = sorted({det[0] for v in gt_ann.values() for det in v})
    pred_global_ids = sorted({det[0] for v in pred_ann.values() for det in v})

    gt_id_to_idx = {gid: i for i, gid in enumerate(gt_global_ids)}
    pred_id_to_idx = {pid: i for i, pid in enumerate(pred_global_ids)}

    for frame in all_frame_ids:
        gt_boxes = gt_ann.get(frame, [])
        pred_boxes = pred_ann.get(frame, [])

        frame_gt_ids = np.array([gt_id_to_idx[g[0]] for g in gt_boxes], dtype=int)
        frame_tracker_ids = np.array([pred_id_to_idx[p[0]] for p in pred_boxes], dtype=int)
        gt_ids.append(frame_gt_ids)
        tracker_ids.append(frame_tracker_ids)

        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            similarity_scores.append(np.zeros((len(gt_boxes), len(pred_boxes))))
            continue

        iou_mat = np.array(compute_iou_matrix(gt_boxes, pred_boxes), dtype=np.float64)
        similarity_scores.append(iou_mat)

    data = {
        "gt_ids": gt_ids,
        "tracker_ids": tracker_ids,
        "similarity_scores": similarity_scores,
        "num_gt_dets": sum(len(v) for v in gt_ann.values()),
        "num_tracker_dets": sum(len(v) for v in pred_ann.values()),
        "num_gt_ids": len(gt_global_ids),
        "num_tracker_ids": len(pred_global_ids),
    }

    return data


def single_video_hota(result_path, gt_path):

    pred_ann = read_annotations(result_path)
    gt_ann = read_annotations(gt_path)

    if pred_ann is None or gt_ann is None:
        return None

    data = build_hota_data(gt_ann, pred_ann)
    hota_metric = HOTA()
    result = hota_metric.eval_sequence(data)
    return result


def multi_video_hota(result_dir, gt_dir):

    print("[evaluation_metrics/hota.py] Computing multi-video HOTA/AssA...")
    result_paths = get_output_files_from_dir(result_dir)
    gt_paths = get_gt_files_from_dir(gt_dir)

    hota_means = []
    assa_means = []

    for video_name, gt_path in gt_paths.items():
        if video_name not in result_paths:
            continue

        pred_path = result_paths[video_name]

        result = single_video_hota(pred_path, gt_path)
        if result is None:
            continue

        # array storing results over alpha thresholds
        hota_arr = result["HOTA"] 
        assa_arr = result["AssA"]

        hota_means.append(np.mean(hota_arr))
        assa_means.append(np.mean(assa_arr))

    if len(hota_means) == 0:
        return {
            "HOTA CI": [0.0, 0.0],
            "AssA CI": [0.0, 0.0],
            "HOTA mean": 0.0,
            "AssA mean": 0.0,
        }

    hota_means = np.array(hota_means)
    assa_means = np.array(assa_means)

    # Bootstrap confidence intervals over mean hota / assa of different videos 
    hota_ci = scipy.stats.bootstrap((hota_means,), np.mean, confidence_level=0.95)
    assa_ci = scipy.stats.bootstrap((assa_means,), np.mean, confidence_level=0.95)

    result_dict = {
        "HOTA CI": [
            float(hota_ci.confidence_interval.low),
            float(hota_ci.confidence_interval.high)
        ],
        "AssA CI": [
            float(assa_ci.confidence_interval.low),
            float(assa_ci.confidence_interval.high)
        ]
    }
    return result_dict


if __name__ == "__main__":
    PROJECT_ROOT = SRC_ROOT.parent
    result_dir = PROJECT_ROOT / "outputs" / "test_pipeline_prob_yolox_x_uncertainty"
    gt_dir = PROJECT_ROOT / "data" / "MOT17" / "train"

    result = multi_video_hota(str(result_dir), str(gt_dir))
    print("HOTA/AssA results:", result)
