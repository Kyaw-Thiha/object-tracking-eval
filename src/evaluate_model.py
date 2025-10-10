import os
import torch
import numpy as np
from PIL import Image
import cv2 as cv

from model.tracker.uncertainty_tracker import UncertaintyTracker
from model.kalman_filter_uncertainty import KalmanFilterWithUncertainty

from datasets.mot17_dataset import MOT17CocoDataset
from torch.utils.data import DataLoader


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("Using device:", device)

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
        images = torch.stack([b[0] for b in batch])      # stack image tensors
        targets = [b[1] for b in batch]                  # keep list of dicts/arrays
        return images, targets
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=detection_collate_fn)
    return dataset, dataloader


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


def main(debug=False):
    # --- Build MOT17 dataset + dataloader ---
    ann_file_path = '/home/allynbao/project/UncertaintyTrack/src/data/MOT17/annotations/half-train_cocoformat.json'
    image_prefix_path = '/home/allynbao/project/UncertaintyTrack/src/data/MOT17/train'

    # --- Get model from factory ---
    from model_factory.opencv_yolox_factory import factory

    model = factory(device=device)
    assert hasattr(model, "get_classes"), "ASSERT ERROR: The model class must have method: get_classes() -> list[str]"
    assert hasattr(model, "infer"), "ASSERT ERROR: The model class must have method: infer() -> tuple[list]"

    class_names = model.get_classes()
    number_classes = len(class_names)

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
        det_score_mode='confidence',
        use_giou=False,
        expand_boxes=True,
        percent=0.3,
        ellipse_filter=True,
        filter_output=True,
        combine_mahalanobis=False,

        primary_cascade={'num_bins': None},   # dict, not False
        secondary_fn=None,
        secondary_cascade={'num_bins': None}, # dict, not False
    )
    motion_model = KalmanFilterWithUncertainty(fps=30)
    tracker.motion = motion_model

    # Wrapper on detector to include motion model, expected by the tracker
    mot_model = MOTDetector(detector=model, motion=KalmanFilterWithUncertainty())

    batch_size = 4
    tracker.reset()

    dataset, dataloader = build_mot17_dataloader(ann_file_path, image_prefix_path, batch_size=batch_size)
   
    current_video = None    # flag to indicate when a new video sequence starts

    results_per_video = []  # saves tracking results for the current video
    output_dir = "./outputs/simplyfied_pipeline_results"
    os.makedirs(output_dir, exist_ok=True)

    # --- Inference ---
    with torch.no_grad():
        for i, (imgs, targets) in enumerate(dataloader):
            imgs = imgs.to(device)     # (B, 3, H, W)

            # --- Inference ---
            batch_dets = mot_model(imgs)

            assert isinstance(batch_dets, tuple) and len(batch_dets) == 3, "Model Inference must return a tuple of (batch_detection_bboxes, batch_detection_labels, batch_detection_bbox_covariance_matrices)"
            
            batch_bboxes, batch_labels, batch_covs = batch_dets
            
            assert isinstance(batch_bboxes, list), "batch bboxes must be a python list"
            assert all(isinstance(t, torch.Tensor) for t in batch_bboxes), \
                "batch_bboxes must contain only torch.Tensor"
            assert isinstance(batch_labels, list), "batch labels must be a python list"
            assert all(isinstance(t, torch.Tensor) for t in batch_labels), \
                "batch_labels must contain only torch.Tensor"
            assert isinstance(batch_covs, list), "batch covs must be a python list"
            assert all(isinstance(t, torch.Tensor) for t in batch_covs), \
                "batch_covs must contain only torch.Tensor"
            
            assert len(batch_bboxes) == len(batch_labels) and len(batch_labels) == len(batch_covs) and len(batch_covs) == batch_size, "result lists lengths must be identical"

            # --- iterate over frames within the batch  ---
            for j in range(batch_size):
                img = imgs[j]
                det_bboxes = batch_bboxes[j]
                det_labels = batch_labels[j]
                bbox_covs = batch_covs[j]

                img_meta = targets[j]['img_metas']
                frame_id = targets[j]['frame_id']
                video_id = targets[j]["video_id"]
                scale_factor = img_meta['scale_factor']

                # scale bboxes back
                scaled_bboxes = det_bboxes.clone()
                scaled_bboxes[:, :4] /= det_bboxes.new_tensor(scale_factor)
                det_bboxes = scaled_bboxes
    
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
                    bboxes=track_bboxes,
                    bbox_covs=track_bbox_covs,
                    labels=track_labels,
                    ids=track_ids,
                    num_classes=number_classes)
           
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
                        tid = int(row[0])       # track ID
                        x1, y1, x2, y2, score = row[1:]
                        w, h = x2 - x1, y2 - y1

                        mot_row = [
                            int(frame_id), tid,
                            float(x1), float(y1), float(w), float(h),
                            float(score),
                            1,   # class id (pedestrian for MOT17)
                            -1   # visibility (not provided)
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
