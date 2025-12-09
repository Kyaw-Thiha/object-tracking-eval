import os

def read_annotation_file(output_path):
    outputs = {}
    with open(output_path, 'r') as f1:
        for line in f1:
            line = line.strip()
            # print(line)
            if len(line.split(",")) == 10:
                frame_id, track_id, x0, y0, x1, y1, score, _, _, _ = line.split(",")
            elif len(line.split(",")) == 9:
                frame_id, track_id, x0, y0, x1, y1, score, _, _ = line.split(",")
            elif len(line.split(",")) == 7:
                frame_id, track_id, x0, y0, x1, y1, score = line.split(",")
            # print(frame_id, track_id, x0, y0, x1, y1, score)
            frame_id = int(float(frame_id))
            track_id = int(float(track_id))
            x0 = float(x0)
            y0 = float(y0)
            x1 = float(x1)
            y1 = float(y1)
            score = float(score)
            if frame_id not in outputs:
                outputs[frame_id] = []
            outputs[frame_id].append((track_id, x0, y0, x1, y1, score))
    return outputs

def read_annotations(output_path):
    if isinstance(output_path, list):
        outputs = {}
        for output_path in output_path:
            new_outputs = read_annotation_file(output_path)
            for frame_id, dets in new_outputs.items():
                outputs.setdefault(frame_id, []).extend(dets)
    else:
        outputs = read_annotation_file(output_path)
    return outputs


def get_output_files_from_dir(output_dir):
    output_paths = {}
    for video_name in os.listdir(output_dir):
        video_name_no_ext = os.path.splitext(video_name)[0]
        output_file = os.path.join(output_dir, video_name)
        if os.path.isfile(output_file):
            output_paths[video_name_no_ext] = output_file
    return output_paths

def get_det_files_from_dir(gt_dir):
    """
    returns a list of det files (detection only, no track ID) under dataset directory (ex. train/ or test/)
    """
    gt_paths = {}
    for video_name in os.listdir(gt_dir):
        video_name_no_ext = os.path.splitext(video_name)[0]
        video_dir = os.path.join(gt_dir, video_name)
        if not os.path.isdir(video_dir):
            continue
        gt_path = os.path.join(video_dir, "det", "det.txt")
        if os.path.isfile(gt_path):
            gt_paths[video_name_no_ext] = gt_path
    return gt_paths


def get_gt_files_from_dir(gt_dir):
    """
    returns a list of gt files under dataset directory (ex. train/ or test/)
    """
    gt_paths = {}
    for video_name in os.listdir(gt_dir):
        video_name_no_ext = os.path.splitext(video_name)[0]
        video_dir = os.path.join(gt_dir, video_name)
        if not os.path.isdir(video_dir):
            continue
        gt_path = os.path.join(video_dir, "gt", "gt.txt")
        if os.path.isfile(gt_path):
            gt_paths[video_name_no_ext] = gt_path
    return gt_paths

def compute_iou_matrix(gt_boxes, det_boxes):
   
    def iou(boxA, boxB):
        # box: (x, y, w, h) -> (x1, y1, x2, y2)
        boxA = (boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3])
        boxB = (boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3])
        
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])

        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 1e-9
        return iou

    iou_matrix = []
    for gt in gt_boxes:
        gt_box = (gt[1], gt[2], gt[3], gt[4])
        row = []
        for det in det_boxes:
            det_box = (det[1], det[2], det[3], det[4])
            row.append(iou(gt_box, det_box))
        iou_matrix.append(row)
    return iou_matrix