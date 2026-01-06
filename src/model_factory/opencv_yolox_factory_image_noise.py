import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_ROOT.parent
YOLOX_ROOT = REPO_ROOT / "object_detection_yolox"
sys.path.append(str(YOLOX_ROOT))
from yolox import YoloX

import cv2 as cv
from PIL import Image

import torch
import numpy as np
import math

class YOLOXNoiseModelWrapper(torch.nn.Module):
    def __init__(self, model, classes, device):
        super().__init__()
        self.model = model
        self.classes = classes
        self.device = device
        self.num_variations = 5
        self.noise_level = 0.1
    
    @torch.inference_mode()
    def infer(self, imgs):
        batch_bboxes = []
        batch_labels = []
        batch_covs = []
        # doesn't support batch inference, convert back to single image at a time
        for i in range(len(imgs)):
            img = imgs[i]
            img_np = img.detach().cpu().numpy().transpose(1, 2, 0)
            print("--new image--")
            print("Preprocess image to add noise")
            processed_imgs_np = self.process_imgs_np(img_np, self.num_variations, self.noise_level)
            # for each image - inference on noised variations of the image
            bboxes = []
            labels = []
            print("Infer on noised variations of the image")
            for processed_img in processed_imgs_np:
                dets = self.model.infer(processed_img)
                print("dets.shape:", dets.shape)
                if len(dets) != 0:
                    det_bboxes = []
                    det_labels = []
                    for det in dets:
                        print("det in dets .shape:", det.shape)
                        box = det[:4]
                        cls_id = int(det[-1])
                        score = det[-2]
                        x0, y0, x1, y1 = box
                        if cls_id == 0:
                            det_bboxes.append([x0, y0, x1, y1, score])
                            det_labels.append(cls_id)
                    bboxes.append(det_bboxes)
                    labels.append(det_labels)
            print("Number of variations with detections:", len(bboxes))
            if not bboxes or not any(len(bb) > 0 for bb in bboxes):
                print("NO Dets this frame")
                batch_bboxes.append(torch.zeros((0, 5), device=self.device))
                batch_labels.append(torch.zeros((0,), device=self.device, dtype=torch.long))
                batch_covs.append(torch.zeros((0, 4, 4), device=self.device))
                continue
            print("Compute mean and covariance of detections across variations")
            
            # process all variations of the detection results to come up with the mean and variance
            mean_bboxes, majority_labels, covariances = self.compute_mean_covariance(bboxes, labels)
            print("mean_bboxes:", mean_bboxes)
            # prepare tensors for tracker
            det_bboxes = torch.tensor(mean_bboxes, dtype=torch.float32, device=self.device)
            print("det_bboxes tensor shape:", det_bboxes.shape)
            det_labels = torch.tensor(majority_labels, dtype=torch.long, device=self.device)
            print("det_labels tensor shape:", det_labels.shape)
            bbox_covs = torch.tensor(covariances, dtype=torch.float32, device=self.device)
            print("bbox_covs tensor shape:", bbox_covs.shape)

            print("correct cov matrices shape: ", torch.eye(4, device=self.device).unsqueeze(0).repeat(det_bboxes.shape[0], 1, 1).shape)

            # --- ChatGPT fix: numpy.linalg.LinAlgError: 1-th leading minor of the array is not positive definite "ensure covariance matrices are SPD before returning" ---
            # error occurs in /home/allynbao/anaconda3/envs/mot_env/lib/python3.10/site-packages/scipy/linalg/decomp_cholesky.py
            sanitized_covs = []
            for cov in bbox_covs:
                # symmetrize to avoid tiny asymmetries
                cov = 0.5 * (cov + cov.T)
                # compute eigenvalues
                eigvals, eigvecs = torch.linalg.eigh(cov)
                # clip tiny / negative eigenvalues
                eps = 1e-3  # enough to guarantee Cholesky works
                eigvals = torch.clamp(eigvals, min=eps)
                # reconstruct SPD matrix
                cov_spd = (eigvecs @ torch.diag(eigvals) @ eigvecs.T)
                sanitized_covs.append(cov_spd)

            bbox_covs = torch.stack(sanitized_covs)

            batch_bboxes.append(det_bboxes)
            batch_labels.append(det_labels)
            batch_covs.append(bbox_covs)

        # convert to tensor
        return (batch_bboxes, batch_labels, batch_covs)
    
    def get_classes(self):
        return self.classes
    
    def process_imgs_np(self, img, num_variations, noise_level):
        variations = []
        for _ in range(num_variations):
            # --- Gaussian noise ---
            sigma = noise_level * np.random.uniform(0.5, 1.5)
            noise = np.random.normal(0, sigma, img.shape)
            noisy_img = np.clip(img + noise, 0, 255)

            variations.append((noisy_img).astype(np.float32))

        return variations
    
    def compute_mean_covariance(self, bboxes, labels, distance_threshold=0.1):
        """
        input:
            bboxes: a list of list of detections for each noised variation of the image
            labels: a list of list of detection labels for each noised variations of the image
        """
        detections_dict = {}
        final_bboxes_mean = []
        final_labels = []
        final_covariances = []
        rolling_ID = 0
        for det_bboxes, det_labels in zip(bboxes, labels):
            if len(detections_dict) == 0:
                for box, label in zip(det_bboxes, det_labels):
                    n = 1 # number of occurence of this detection accross variations of the img
                    raw_bboxes = [box] # will keep track of the same bboxes across variation of the img
                    detections_dict[rolling_ID] = (box, [label], n, raw_bboxes) 
                    rolling_ID += 1
            else:
                for box, label in zip(det_bboxes, det_labels):
                    # find closest match from existing detections in dict
                    min_distance = float("Inf")
                    min_ID = None
                    for ID in detections_dict.keys():
                        reference_bbox, _, _, _ = detections_dict[ID]
                        distance = self.get_distance(box, reference_bbox)
                        if distance < distance_threshold and distance < min_distance:
                            min_distance = distance
                            min_ID = ID
                    
                    if min_ID is not None:
                        old_bbox, old_labels, n, raw_bboxes = detections_dict[min_ID]
                        old_x0, old_y0, old_x1, old_y1, old_score = old_bbox
                        x0, y0, x1, y1, score = box
                        # compute an average
                        new_x0 = n/(n+1) * old_x0 + 1/(n+1) * x0
                        new_y0 = n/(n+1) * old_y0 + 1/(n+1) * y0
                        new_x1 = n/(n+1) * old_x1 + 1/(n+1) * x1
                        new_y1 = n/(n+1) * old_y1 + 1/(n+1) * y1

                        # average score
                        new_score = n/(n+1) * old_score + 1/(n+1) * score
                        
                        new_avg_box = (new_x0, new_y0, new_x1, new_y1, new_score)
                        old_labels.append(label)
                        n += 1
                        raw_bboxes.append(box)

                        detections_dict[min_ID] = (new_avg_box, old_labels, n, raw_bboxes)

                    else:
                        n = 1 # number of occurence of this detection accross variations of the img
                        raw_bboxes = [box] # will keep track of the same bboxes across variation of the img
                        detections_dict[rolling_ID] = (box, [label], n, raw_bboxes) 
                        rolling_ID += 1
        
        # compute covariance & most common label
        for ID in detections_dict.keys():
            avg_box, labels, n, raw_bboxes = detections_dict[ID]
            
            # covariance
            coords = np.array(raw_bboxes, dtype=np.float32)
            if len(coords) > 1:
                coords_xyxy = coords[:, :4]
                covariance = np.cov(coords_xyxy.T)  # shape (4,4)

                diag = np.diag(covariance)
                if np.any(diag <= 0):
                    print("Cov is negative!!")

                print("computed cov:", covariance[0][0])
            else:
                covariance = np.zeros((4,4))

            most_common_label = max(set(labels), key=labels.count)
            final_bboxes_mean.append(avg_box)
            final_labels.append(most_common_label)
            final_covariances.append(covariance)
        
        return final_bboxes_mean, final_labels, final_covariances
    
    def get_distance(self, bbox1, bbox2):
        # calcualte euclidian distance between two bboxes
        x1_0, y1_0, x1_1, y1_1, _ = bbox1
        x2_0, y2_0, x2_1, y2_1, _ = bbox2

        top_left_dist = math.sqrt((x1_0 - x2_0)**2 + (y1_0 - y2_0)**2)
        bottom_right_dist = math.sqrt((x1_1 - x2_1)**2 + (y1_1 - y2_1)**2)

        avg_distance = 1/2 * (top_left_dist + bottom_right_dist)
        return avg_distance


        

def factory(device):
    model_checkpoint_path = YOLOX_ROOT / "object_detection_yolox_2022nov.onnx"

    # Check OpenCV version
    opencv_python_version = lambda str_version: tuple(map(int, (str_version.split("."))))
    assert opencv_python_version(cv.__version__) >= opencv_python_version("4.10.0"), \
        "Please install latest opencv-python for benchmark: python3 -m pip install --upgrade opencv-python"

    # Valid combinations of backends and targets
    backend_target_pairs = [
        [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
        [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
        [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
        [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
        [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
    ]

    backend_target = 0
    backend_id = backend_target_pairs[backend_target][0]
    target_id = backend_target_pairs[backend_target][1]

    # --- Initialize modle for inference ---
    class_names = ('person', 'bicycle', 'car', 'motorcycle', 'ai   rplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

    model = YoloX(modelPath=str(model_checkpoint_path),
                      confThreshold=0.5,
                      nmsThreshold=0.5,
                      objThreshold=0.5,
                      backendId=backend_id,
                      targetId=target_id)
    
    final_model = YOLOXNoiseModelWrapper(model, class_names, device)
    
    return final_model
