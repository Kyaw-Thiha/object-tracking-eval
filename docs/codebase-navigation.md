# Codebase Navigation

As the pipeline uses existing implementations for object tracking algorithms and the probabilistic detector from the UncertaintyTrack repository [Lee et al. 2024], this repository is a fork of UncertaintyTrack. 

The codebase can be found on GitHub at: GitHub - 
[Allyn-Bao/UncertaintyTrack: Tweaked UncertaintyTrack as a pipeline for model evaluation](https://github.com/Allyn-Bao/UncertaintyTrack)
## Pipeline

The main multi-object tracking evaluation pipeline is implemented as a Python script with the following capabilities:

1. Argument parser for configuring the dataset, data loader, model factory, and object-tracker selections, as well as specifying output directories for evaluation results and annotation files.
2. Dynamic import and initialization of the data loader, detector model, and object tracker.
3. Main inference loop where object detection and tracking run on batches of test data.
4. Runtime measurement and per-sequence throughput reporting.
5. Computation of DetA, AssA, and HOTA metrics on the recorded inference results.
6. Saving inference outputs and evaluation metrics to disk.

The pipeline script lives at `src/evaluation_pipeline.py`, and the accompanying shell wrapper (`src/evaluation_pipeline.sh`) provides the command used to execute it.

## Dataset & DataLoader Factory

COCO-style datasets should be saved under the root `data/` directory. The pipeline uses PyTorch datasets and data loaders, so two Python scripts are required: one to define the dataset and another to build the loader. When a new dataset already follows the COCO-style format, the existing CAMEL or MOT17 implementations can be adapted with minimal changes.

1. **Dataset class**: a script under `src/datasets/` defining a subclass of `torch.utils.data.Dataset` that implements `__init__()`, `__len__()`, and `__getitem__()`. Use the provided CAMEL and MOT17 dataset classes (both COCO-style) as references for new datasets.
2. **Data-loader factory**: a script under `src/dataloader_factory/` exposing `factory()` functions that instantiate the dataset class and return a `torch.utils.data.DataLoader`. Use the provided CAMEL and MOT17 factories as references for new dataloader.

## Detector Model Factory

To let the pipeline run arbitrary detector models (including different covariance-estimation approaches), a shared interface mirrors the data-loader factory pattern. Each detector must provide a factory script so the pipeline can initialize and query the model consistently.

- Create a Python module under `src/model_factory/` that exposes a `factory()` function returning a subclass of `torch.nn.Module`.
- The model object must implement:
  1. `infer(torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]`, which receives a batch of image tensors shaped `(batch, 3, H, W)` and returns three lists (one per image) for bounding boxes, class labels, and covariance matrices. For an image with `N` detections: bounding boxes are shaped `(N, 5)` (`[x, y, width, height, confidence]`), labels are `(N,)`, and covariances are `(N, 4, 4)`.
  2. `get_classes() -> List[str]`, returning the class names used by the model.

- Reference implementations (YOLOX-based, [Lee et al. 2024]):
  - `model_factory/yolox_identity_covs.py`: deterministic covariance
  - `model_factory/yolox_noise.py`: test-time augmentation
  - `model_factory/prob_yolox.py`: probabilistic covariance

## Probabilistic YOLOX

For the probabilistic YOLOX detector, the model head lives under `src/model/det/yolox/`. The training configuration with all hyperparameters is `src/configs/yolox/prob_yolox_x_es_mot17-half.py`, and the corresponding training entry point is `src/train.py`.

## Object Trackers

All object trackers used by the pipeline are the original UncertaintyTrack implementations found in `src/model/tracker/`.

## Evaluation Metrics

Metric computation lives in `src/evaluation_metrics/evaluate.py`, which consumes the ground-truth annotations and predicted tracks. DetA-specific logic is implemented in `src/evaluation_metrics/detA.py`. We reuse `TrackEval Library` for HOTA and AssA, while DetA reports detection-accuracy confidence intervals across all frames and HOTA/AssA report tracking/association confidence intervals across all sequences in the test set.

## Plotting Results

`src/plot_evaluation_results.py` generates comparative bar charts for a curated set of evaluation outputs. These plotting scripts are narrow in scope, so new experiments typically require authoring task-specific plotting utilities.

