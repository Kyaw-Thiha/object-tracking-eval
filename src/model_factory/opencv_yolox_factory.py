import sys
sys.path.append("/home/allynbao/project/object_detection_yolox")
from yolox import YoloX

import cv2 as cv
from PIL import Image

import torch

class YOLOXModelWrapper(torch.nn.Module):
    def __init__(self, model, classes, device):
        super().__init__()
        self.model = model
        self.classes = classes
        self.device = device
    
    @torch.inference_mode()
    def infer(self, imgs):
        batch_bboxes = []
        batch_labels = []
        batch_covs = []
        # doesn't support batch inference, convert back to single image at a time
        for i in range(len(imgs)):
            img = imgs[i]
            img_np = img.detach().cpu().numpy().transpose(1, 2, 0)
            dets = self.model.infer(img_np)
            det_bboxes = []
            det_labels = []
            for det in dets:
                box = det[:4]
                cls_id = int(det[-1])
                score = det[-2]
                x0, y0, x1, y1 = box
                if cls_id == 0:
                    det_bboxes.append([x0, y0, x1, y1, score])
                    det_labels.append(cls_id)
        
            # prepare tensors for tracker
            det_bboxes = torch.tensor(det_bboxes, dtype=torch.float32, device=self.device)
            det_labels = torch.tensor(det_labels, dtype=torch.long, device=self.device)
            bbox_covs = torch.eye(4, device=img.device).unsqueeze(0).repeat(det_bboxes.shape[0], 1, 1)

            batch_bboxes.append(det_bboxes)
            batch_labels.append(det_labels)
            batch_covs.append(bbox_covs)

        # convert to tensor
        return (batch_bboxes, batch_labels, batch_covs)
    
    def get_classes(self):
        return self.classes
        

def factory(device):

    model_checkpoint_path = "/home/allynbao/project/object_detection_yolox/object_detection_yolox_2022nov.onnx"

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
    class_names = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
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

    model = YoloX(modelPath=model_checkpoint_path,
                      confThreshold=0.5,
                      nmsThreshold=0.5,
                      objThreshold=0.5,
                      backendId=backend_id,
                      targetId=target_id)
    
    final_model = YOLOXModelWrapper(model, class_names, device)
    
    return final_model