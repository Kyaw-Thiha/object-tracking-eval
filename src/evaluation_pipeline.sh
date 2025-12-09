#!/bin/bash

DEVICE="cuda"
DATASET="camel"
# TRACKER="uncertainty_tracker"
TRACKER="probabilistic_byte_tracker"
# TRACKER="prob_ocsort_tracker"

# MODEL_FACTORY="yolox_identity_covs"
MODEL_FACTORY="yolox_noise"
# MODEL_FACTORY="prob_yolox"

python -X faulthandler evaluation_pipeline.py \
    --dataloader_factory "${DATASET}_factory.py" \
    --dataset_dir "/home/allynbao/project/UncertaintyTrack/src/data/${DATASET}_dataset/test/" \
    --model_factory "${MODEL_FACTORY}.py" \
    --tracker "${TRACKER}" \
    --device "${DEVICE}" \
    --output_dir "/home/allynbao/project/UncertaintyTrack/src/outputs/${DATASET}_${MODEL_FACTORY}_${TRACKER}/" \
    --eval_result_dir "/home/allynbao/project/UncertaintyTrack/src/evaluation_results/${DATASET}_${MODEL_FACTORY}_${TRACKER}/" \
    --plot_save_path "/home/allynbao/project/UncertaintyTrack/src/plots/${DATASET}_${MODEL_FACTORY}_${TRACKER}/" \
