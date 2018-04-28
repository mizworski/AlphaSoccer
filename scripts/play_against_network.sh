#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="resnet_v1"
LOG_DIR=models/logs/${JOB_NAME}${now}
MODEL_DIR=models/${JOB_NAME}
#MODEL_DIR=models/smaller_board
REPLAY_DIR=data/replays/${JOB_NAME}

python3 alphasoccer/environment/play_against_network.py \
    --model_dir ${MODEL_DIR} \
    --n_rollouts 1600 \
    --c_puct 5.0 \
    --n_kernels 96 \
    --residual_blocks 7 \
    --moves_before_decaying 2 \
    --initial_temperature 0.2