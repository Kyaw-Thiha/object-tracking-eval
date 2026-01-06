# Project Structure

```
object-tracking-eval/
├── docker/
│   └── ...
├── docs/
│   ├── codebase_navigation.md
│   ├── dataset_format.md
│   ├── evaluation_metrics.md
│   └── ...
├── src/
│   ├── configs/
│   │   └── ...
│   ├── core/
│   │   └── ...
│   ├── dataloader_factory/
│   │   └── ...
│   ├── datasets/
│   │   └── ...
│   ├── evaluation_metrics/
│   │   ├── AssA.py
│   │   ├── detA.py
│   │   ├── hota.py
│   │   ├── evaluate.py
│   │   └── ...
│   ├── model/
│   │   ├── det/
│   │   │   └── yolox/
│   │   └── tracker/
│   │       └── ...
│   ├── model_factory/
│   │   ├── prob_yolox.py
│   │   └── ...
│   ├── evaluation_pipeline.py
│   ├── evaluation_pipeline.sh
│   ├── plot_evaluation_results.py
│   ├── train.py
│   └── ...
├── README.md
└── LICENSE
```

- `docs/`: reference material covering dataset format, evaluation metrics, and navigation guidance.
- `src/evaluation_pipeline.py`: main multi-object tracking evaluation driver (arg parsing, dynamic loading, inference, metrics).
- `src/evaluation_pipeline.sh`: convenience script that wraps the Python pipeline invocation.
- `src/datasets/`: PyTorch dataset definitions (COCO-style datasets for CAMEL, MOT17, etc.).
- `src/dataloader_factory/`: factories that instantiate datasets/data loaders so the pipeline can dynamically select datasets.
- `src/model_factory/`: detector factory modules (YOLOX variants) that expose `factory()` and `infer()` interfaces.
- `src/model/tracker/`: tracker implementations reused from UncertaintyTrack.
- `src/evaluation_metrics/`: logic for DetA, AssA, HOTA, and evaluation orchestration.
- `src/configs/`: detector/training configuration files (e.g., `prob_yolox_x_es_mot17-half.py`).
- `src/plot_evaluation_results.py`: helper for plotting comparative evaluation bar charts.
- `docker/`: container scripts/assets for reproducible environments (usually untouched unless rebuilding images).

