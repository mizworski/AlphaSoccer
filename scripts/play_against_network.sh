#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="resnet_standard_size"
LOG_DIR=models/logs/${JOB_NAME}${now}
MODEL_DIR=models/${JOB_NAME}
#MODEL_DIR=models/smaller_board
REPLAY_DIR=data/replays/${JOB_NAME}

python3 alphasoccer/environment/play_against_network.py \
    --model_dir ${MODEL_DIR} \
    --n_rollouts 1600 \
    --c_puct 10.0 \
    --n_kernels 128 \
    --residual_blocks 8 \
    --moves_before_decaying 2 \
    --initial_temperature 0.3