# UncertaintyTrack
Pipeline for MOT model evaluation built on UncertaintyTrack repository

# Using Simplified Inference Pipeline for model evaluation
the inference script is under src/evaluate_model.py

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


# Using UncertaintyTrack Legacy train/test scripts
## Training:
example
```
python3.10 train.py configs/bytetrack/bytetrack_yolox_x_3x6_mot17-half.py \
    --work-dir ./work_dirs/test_run --no-validate \
    --cfg-options data.workers_per_gpu=1 \
    model.detector.init_cfg.checkpoint=checkpoints/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth
```

## Test / evaluation:
### Eval mode: outputs evaluation metrics
example
```
nohup python3.10 test.py configs/bytetrack/bytetrack_yolox_x_3x6_mot17-half.py \
    --checkpoint work_dirs/test_run/latest.pth \
    --eval track \
    --out results.pkl > nohup_test_added_test_dataset_in_config.out
```

### Formate only mode: outputs annotation files from inference as well as annotated videos
example
```
nohup python3.10 test.py configs/bytetrack/bytetrack_yolox_x_3x6_mot17-half.py \
    --checkpoint work_dirs/test_run/latest.pth \
    --out results.pkl \
    --format-only > nohup_test_full_test_dataset.out
```

## Custom config File with fixed covariance value
path: configs/custom/pseudo_uncertainmot_yolox_x_3x6_mot17-half.py
command to run test.py:
```
nohup python3.10 test.py configs/custom/pseudo_uncertainmot_yolox_x_3x6_mot17-half.py \
    --checkpoint work_dirs/test_run/latest.pth \
    --out results_custom.pkl \
    --format-only > nohup_custom.out
```