# Quickstart Guideline for testing Prob YOLOX

## Prerequisite
1. Dataset: `camel_dataset` (in COCO format)
2. Model Checkpoint: `epoch_26.pth` (can be different checkpoint in the future)

## Folder Structure
1. Put the dataset in `data/camel_dataset` .
2. Put the checkpoint in `checkpoints/prob_yolox_camel/epoch_26.pth`

## Running the evaluation pipeline
```bash
python -X faulthandler src/evaluation_pipeline.py \
    --dataloader_factory camel_factory.py \
    --dataset_dir data/camel_dataset/test \
    --model_factory prob_yolox.py \
    --tracker probabilistic_byte_tracker \
    --device cuda \
    --output_dir outputs/camel_prob_yolox_probabilistic_byte_tracker \
    --eval_result_dir evaluation_results/camel_prob_yolox_probabilistic_byte_tracker \
    --plot_save_path plots/camel_prob_yolox_probabilistic_byte_tracker
```


