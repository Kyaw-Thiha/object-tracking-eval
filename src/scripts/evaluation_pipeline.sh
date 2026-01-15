#!/bin/bash

DEVICE="cuda"
DATASET="camel"
# TRACKER="uncertainty_tracker"
TRACKER="probabilistic_byte_tracker"
# TRACKER="prob_ocsort_tracker"

# MODEL_FACTORY="yolox_identity_covs"
MODEL_FACTORY="yolox_noise"
# MODEL_FACTORY="prob_yolox"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$SRC_ROOT/.." && pwd)"

# Allow callers to override the default roots to avoid hard-coded absolute paths.
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"
EVAL_RESULT_ROOT="${EVAL_RESULT_ROOT:-$PROJECT_ROOT/evaluation_results}"
PLOT_ROOT="${PLOT_ROOT:-$PROJECT_ROOT/plots}"

DATASET_DIR="${DATA_ROOT}/${DATASET}_dataset/test"
OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}_${MODEL_FACTORY}_${TRACKER}"
EVAL_RESULT_DIR="${EVAL_RESULT_ROOT}/${DATASET}_${MODEL_FACTORY}_${TRACKER}"
PLOT_SAVE_PATH="${PLOT_ROOT}/${DATASET}_${MODEL_FACTORY}_${TRACKER}"

mkdir -p "$OUTPUT_DIR" "$EVAL_RESULT_DIR" "$PLOT_SAVE_PATH"

python -X faulthandler "$SRC_ROOT/evaluation_pipeline.py" \
    --dataloader_factory "${DATASET}_factory.py" \
    --dataset_dir "$DATASET_DIR" \
    --model_factory "${MODEL_FACTORY}.py" \
    --tracker "${TRACKER}" \
    --device "${DEVICE}" \
    --output_dir "$OUTPUT_DIR" \
    --eval_result_dir "$EVAL_RESULT_DIR" \
    --plot_save_path "$PLOT_SAVE_PATH" \
