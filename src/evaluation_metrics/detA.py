import os
from evaluation_metrics.utils import (
    read_annotations, 
    get_det_files_from_dir, 
    get_output_files_from_dir, 
    compute_iou_matrix, 
)

import numpy as np
import scipy

def compute_detA_single_frame(gt_boxes, det_boxes, iou_matrix, iou_threshold=0.5):
    """
    gt_boxes: list of GT bboxes for the frame
    det_boxes: list of det bboxes for the frame
    iou_matrix: computed IoU matrix [num_gt x num_det]
    """

    if len(gt_boxes) == 0 and len(det_boxes) == 0:
        return 1.0  # empty frame, perfect by definition

    if len(gt_boxes) == 0 or len(det_boxes) == 0:
        return 0.0  # no matches possible

    # Convert IoU to cost matrix
    cost = 1 - np.array(iou_matrix)

    # Use Hungarian algorithm
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)

    # Filter matches by IoU threshold
    matched_pairs = []
    for r, c in zip(row_ind, col_ind):
        iou_val = iou_matrix[r][c]
        if iou_val >= iou_threshold:
            matched_pairs.append(iou_val)

    if len(matched_pairs) == 0:
        return 0.0

    detA_frame = sum(matched_pairs) / len(matched_pairs)
    return detA_frame

def single_video_detA(result_path, gt_path) -> list:
    """
    Compute average DetA metric for a single video, returns list of DetA over frames (used for conference interval later)
    """
    results = read_annotations(result_path)
    gts = read_annotations(gt_path)

    detA_over_frames = []

    no_det_frames = []
    for frame_id in gts.keys():
        # no detections in this frame
        if frame_id not in results.keys():
            no_det_frames.append(frame_id)

        # compute IoU matrix for current frame
        gt_boxes = gts[frame_id]
        det_boxes = results.get(frame_id, [])
        if det_boxes:
            iou_matrix = compute_iou_matrix(gt_boxes, det_boxes)
            avg_detA = compute_detA_single_frame(gt_boxes, det_boxes, iou_matrix)
            detA_over_frames.append(avg_detA)
        else:
            # TODO: Currently skipping frames with no detections, as sometimes we only inference part of the dataset
            pass
    
    return detA_over_frames


def multi_video_detA(result_dir_path, det_dir_path) -> dict:
    """
    Compute average DetA metric for multiple videos, returns list of DetA over frames (used for conference interval later)
    """
    print("[evaluation_metrics/detA.py] Computing multi-video DetA...")
    result_paths = get_output_files_from_dir(result_dir_path)
    det_paths = get_det_files_from_dir(det_dir_path)

    all_detA_over_frames = []

    for video_name in det_paths.keys():
        if video_name in result_paths.keys():
            result_path = result_paths[video_name]
            det_path = det_paths[video_name]
            detA_over_frames = single_video_detA(result_path, det_path)
            all_detA_over_frames.extend(detA_over_frames)

    # compute conference interval (scipy.stats.bootstrap)
    res = scipy.stats.bootstrap((np.array(all_detA_over_frames),), np.mean, confidence_level=0.95)
    # print(f"DetA 95% confidence interval over all videos: {res.confidence_interval}, mean: {res.standard_error + np.mean(all_detA_over_frames)}")

    detA_ci = {"DetA CI": [float(res.confidence_interval.low), float(res.confidence_interval.high)]}
    return detA_ci


if __name__ == "__main__":

    RESULT_DIR = "/home/allynbao/project/UncertaintyTrack/src/outputs/test_pipeline_prob_yolox_x_uncertainty"
    GT_DIR = "/home/allynbao/project/UncertaintyTrack/src/data/MOT17/train"

    result = multi_video_detA(RESULT_DIR, GT_DIR)

    print("DetA CI over all videos:", result)