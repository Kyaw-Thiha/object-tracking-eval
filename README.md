# UncertaintyTrack
Pipeline for MOT model evaluation built on UncertaintyTrack repository

## Getting Started
Note that the python version needs to be 3.10 due to `mmcv-full` package version.

1. Create the virtual environment.
```bash
conda create -n object-tracking-eval python=3.10 -y
```

2. Activate the virtual environment.
```bash
conda activate object-tracking-eval
```

3. Install the specific torch versions.
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

4. Install the `mmcv-full` package.
```bash
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
```

5. Install the rest of the packages
```bash
pip install -r requirements.txt
```

# Using Simplified Inference Pipeline for model evaluation
The inference script is `src/evaluation_pipeline.py`.

## Detection Model requirements
Assure a model builder script is saved under src/model_factory, which must contain function factory(torch.device), which returns the model instance.
Detection model must accept batch image tensor input in the dimention of (B, 3, H, W). input image tensors are already rescaled to 640x640.
Model instance must have a class method model.infer(imgs).
model.infer(imgs) must return detection result in a tuple of 3 elements, it is expected:
```
batch_bboxes, batch_labels, batch_covs = mot_model(imgs)
batch_bboxes    # list[torch.tensor([num_detections, 5])]       each row [x1, y1, x2, y2, score]
batch_labels    # list[torch.tensor([num_detections])]
batch_covs      # list[torch.tensor([num_detections, 4, 4])]
```


# Training (custom pipeline)
Use `src/train.py` with a MMDet/MMTrack-style config (as model zoo only) and a
dataloader factory from `src/data/dataloaders/`.

Example:
```
python3.10 src/train.py \
    --config src/configs/yolox/prob_yolox_x_es_mot17-half.py \
    --checkpoint checkpoints/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth \
    --work-dir work_dirs/prob_yolox_x_es_mot17_half \
    --dataloader-factory mot17_train_dataloader \
    --epochs 80 \
    --lr 1e-4 \
    --batch-size 4
```
