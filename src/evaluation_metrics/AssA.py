import os
from pathlib import Path
import numpy as np
import scipy.stats
from scipy.optimize import linear_sum_assignment

from evaluation_metrics.utils import (
    read_annotations,
    get_gt_files_from_dir,
    get_output_files_from_dir,
    compute_iou_matrix,
)

SRC_ROOT = Path(__file__).resolve().parents[1]


def build_global_id_maps(gts, results):
    """
    Build global index maps for GT track IDs and predicted track IDs.

    Args:
        gts: dict[frame_id -> list[(gt_id, x0, y0, x1, y1, score)]]
        results: dict[frame_id -> list[(pred_id, x0, y0, x1, y1, score)]]

    Returns:
        gt_id_to_idx, pred_id_to_idx
    """
    gt_id_to_idx = {}
    pred_id_to_idx = {}

    # Collect GT IDs
    for frame_id, gt_boxes in gts.items():
        for gt in gt_boxes:
            gt_id = int(gt[0])
            if gt_id not in gt_id_to_idx:
                gt_id_to_idx[gt_id] = len(gt_id_to_idx)

    # Collect predicted IDs
    for frame_id, det_boxes in results.items():
        for det in det_boxes:
            pred_id = int(det[0])
            if pred_id not in pred_id_to_idx:
                pred_id_to_idx[pred_id] = len(pred_id_to_idx)

    return gt_id_to_idx, pred_id_to_idx


def compute_global_alignment(gts, results, gt_id_to_idx, pred_id_to_idx):
    """
    Pass 1: build the global alignment matrix A_ij ~ Jaccard-like
    over track pairs, following your pseudocode:

        potential_matches += normalized_similarity(frame_similarity)
        global_alignment = potential_matches / (gt_count + pred_count - potential_matches)

    Here:
        - similarity = IoU (already in [0,1])
        - gt_count[i] = number of frames where GT track i appears
        - pred_count[j] = number of frames where prediction track j appears

    Returns:
        global_alignment: np.ndarray [N_gt, N_pred]
    """
    num_gt = len(gt_id_to_idx)
    num_pred = len(pred_id_to_idx)

    # Potential matches aggregated over time
    potential_matches = np.zeros((num_gt, num_pred), dtype=np.float64)

    # Track "length" in frames
    gt_count = np.zeros(num_gt, dtype=np.float64)
    pred_count = np.zeros(num_pred, dtype=np.float64)

    # First, count frames per track
    for frame_id, gt_boxes in gts.items():
        for gt in gt_boxes:
            gt_idx = gt_id_to_idx[int(gt[0])]
            gt_count[gt_idx] += 1

    for frame_id, det_boxes in results.items():
        for det in det_boxes:
            pred_idx = pred_id_to_idx[int(det[0])]
            pred_count[pred_idx] += 1

    # Now accumulate IoU similarities across frames
    # Only frames where both GT and detections exist contribute
    common_frame_ids = sorted(set(gts.keys()) & set(results.keys()))
    for frame_id in common_frame_ids:
        gt_boxes = gts[frame_id]
        det_boxes = results[frame_id]

        if not gt_boxes or not det_boxes:
            continue

        # IoU matrix [num_gt_frame x num_pred_frame]
        iou_matrix = compute_iou_matrix(gt_boxes, det_boxes)
        iou_matrix = np.array(iou_matrix, dtype=np.float64)

        # Map local ids in this frame to global indices
        gt_ids_local = [int(gt[0]) for gt in gt_boxes]
        pred_ids_local = [int(det[0]) for det in det_boxes]

        for i, gt_id in enumerate(gt_ids_local):
            gi = gt_id_to_idx[gt_id]
            for j, pred_id in enumerate(pred_ids_local):
                pj = pred_id_to_idx[pred_id]
                sim = iou_matrix[i, j]  # already in [0,1]
                potential_matches[gi, pj] += sim

    # Jaccard-like global alignment
    # A_ij = potential_matches_ij / (gt_count_i + pred_count_j - potential_matches_ij)
    # Handle division by 0 gracefully.
    global_alignment = np.zeros_like(potential_matches)
    for gi in range(num_gt):
        for pj in range(num_pred):
            denom = gt_count[gi] + pred_count[pj] - potential_matches[gi, pj]
            if denom > 0:
                global_alignment[gi, pj] = potential_matches[gi, pj] / denom
            else:
                global_alignment[gi, pj] = 0.0

    return global_alignment


def single_video_AssA(result_path, gt_path, alphas=None):
    """
    Compute HOTA-style AssA for a single video, following your 2-pass
    global alignment + per-frame Hungarian scheme.

    Args:
        result_path: path to tracker output (MOT-style text: frame, id, x0,y0,x1,y1,score,...)
        gt_path: path to ground truth GT file (gt/gt.txt with IDs)
        alphas: iterable of IoU thresholds in [0,1]. If None, use 0.05..0.95.

    Returns:
        scalar AssA (average over alphas)
    """
    if alphas is None:
        # A modest set of thresholds for HOTA-style integration
        alphas = np.linspace(0.05, 0.95, 10)

    # Frame -> list[(id, x0, y0, x1, y1, score)]
    results = read_annotations(result_path)
    gts = read_annotations(gt_path)

    if results is None or gts is None:
        # Defensive: if parsing failed, return 0
        return 0.0

    # Build global index maps
    gt_id_to_idx, pred_id_to_idx = build_global_id_maps(gts, results)

    # If there are no tracks at all, define AssA = 0
    if len(gt_id_to_idx) == 0 or len(pred_id_to_idx) == 0:
        return 0.0

    # Pass 1: global alignment A_ij
    global_alignment = compute_global_alignment(gts, results,
                                                gt_id_to_idx, pred_id_to_idx)

    # Pass 2: per-frame matching with re-weighted similarity
    # We'll accumulate TP/FP/FN for each alpha
    AssA_per_alpha = []

    # Pre-sort frame IDs for consistent iteration
    all_frame_ids = sorted(set(gts.keys()) | set(results.keys()))

    for alpha in alphas:
        TP = 0
        FP = 0
        FN = 0

        for frame_id in all_frame_ids:
            gt_boxes = gts.get(frame_id, [])
            det_boxes = results.get(frame_id, [])

            num_gt_f = len(gt_boxes)
            num_det_f = len(det_boxes)

            # No objects in this frame
            if num_gt_f == 0 and num_det_f == 0:
                continue

            # If one side is empty: everything is FN or FP
            if num_gt_f == 0 and num_det_f > 0:
                FP += num_det_f
                continue
            if num_gt_f > 0 and num_det_f == 0:
                FN += num_gt_f
                continue

            # Compute IoU similarity matrix for this frame
            iou_matrix = compute_iou_matrix(gt_boxes, det_boxes)
            iou_matrix = np.array(iou_matrix, dtype=np.float64)

            # Map to global alignment for these tracks
            gt_ids_local = [int(gt[0]) for gt in gt_boxes]
            pred_ids_local = [int(det[0]) for det in det_boxes]

            align_sub = np.zeros_like(iou_matrix)
            for i, gt_id in enumerate(gt_ids_local):
                gi = gt_id_to_idx.get(gt_id, None)
                if gi is None:
                    continue
                for j, pred_id in enumerate(pred_ids_local):
                    pj = pred_id_to_idx.get(pred_id, None)
                    if pj is None:
                        continue
                    align_sub[i, j] = global_alignment[gi, pj]

            # Frame-wise combined score
            frame_score = align_sub * iou_matrix  # shape (num_gt_f, num_det_f)

            # If everything is zero, matching is pointless → pure FP/FN
            if np.all(frame_score == 0):
                FN += num_gt_f
                FP += num_det_f
                continue

            # Hungarian wants a cost to minimize → use cost = 1 - normalized_score
            # Clamp scores to [0,1] just in case
            frame_score_clamped = np.clip(frame_score, 0.0, 1.0)
            cost = 1.0 - frame_score_clamped

            row_ind, col_ind = linear_sum_assignment(cost)

            matched_gt_indices = set()
            matched_det_indices = set()

            for r, c in zip(row_ind, col_ind):
                # Apply IoU threshold on raw IoU, not re-weighted score
                if iou_matrix[r, c] >= alpha and frame_score_clamped[r, c] > 0:
                    matched_gt_indices.add(r)
                    matched_det_indices.add(c)

            # Count TP, FP, FN for this frame at this alpha
            TP += len(matched_gt_indices)
            FN += (num_gt_f - len(matched_gt_indices))
            FP += (num_det_f - len(matched_det_indices))

        denom = TP + FP + FN
        if denom > 0:
            AssA_alpha = TP / denom
        else:
            AssA_alpha = 0.0

        AssA_per_alpha.append(AssA_alpha)

    # Final AssA: mean over alpha thresholds
    if len(AssA_per_alpha) == 0:
        return 0.0

    return float(np.mean(AssA_per_alpha))


def multi_video_AssA(result_dir_path, gt_dir_path):
    """
    Compute HOTA-style AssA over multiple videos, and return a
    95% bootstrap confidence interval over per-video AssA scores.

    Args:
        result_dir_path: directory with MOT-style tracker outputs
                         (one txt per sequence).
        gt_dir_path: MOT dataset root (e.g. MOT17/train), containing
                     subdirs with 'gt/gt.txt'.

    Returns:
        dict like {"AssA CI": [low, high], "AssA mean": mean}
    """
    result_paths = get_output_files_from_dir(result_dir_path)
    gt_paths = get_gt_files_from_dir(gt_dir_path)

    per_video_AssA = []

    for video_name, gt_path in gt_paths.items():
        if video_name not in result_paths:
            # No result for this video → skip
            continue

        result_path = result_paths[video_name]
        print(f"[INFO] Computing AssA for video: {video_name}")
        assa_val = single_video_AssA(result_path, gt_path)
        per_video_AssA.append(assa_val)

    if len(per_video_AssA) == 0:
        print("[WARN] No videos with both GT and results found. AssA is undefined.")
        return {"AssA CI": [0.0, 0.0], "AssA mean": 0.0}

    per_video_AssA = np.array(per_video_AssA, dtype=np.float64)
    mean_AssA = float(np.mean(per_video_AssA))

    # Bootstrap CI over videos
    res = scipy.stats.bootstrap(
        (per_video_AssA,),
        np.mean,
        confidence_level=0.95,
        method="basic"
    )

    low = float(res.confidence_interval.low)
    high = float(res.confidence_interval.high)

    print(f"AssA 95% confidence interval over videos: [{low:.4f}, {high:.4f}], "
          f"mean: {mean_AssA:.4f}")

    return {
        "AssA CI": [low, high],
        "AssA mean": mean_AssA,
    }


if __name__ == "__main__":
    # Example usage (adapt paths as needed)
    result_dir = SRC_ROOT / "outputs" / "test_pipeline_prob_yolox_x_uncertainty"
    gt_dir = SRC_ROOT / "data" / "MOT17" / "train"

    result = multi_video_AssA(str(result_dir), str(gt_dir))
    print("AssA CI over all videos:", result)
