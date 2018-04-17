#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="resnet"
LOG_DIR=models/logs/${JOB_NAME}_${now}
MODEL_DIR=models/${JOB_NAME}_${now}
#MODEL_DIR=models/${JOB_NAME}
REPLAY_DIR=data/replays/${JOB_NAME}

python3 alphasoccer/actor_critic/run_soccer.py \
    --model_dir ${MODEL_DIR} \
    --log_dir ${LOG_DIR} \
    --replay_dir ${REPLAY_DIR} \
    --n_total_timesteps 100 \
    --n_self_play_games 0 \
    --n_evaluation_games 0 \
    --learning_rate 1e-3 \
    --n_evaluations 10 \
    --n_training_steps 1024 \
    --batch_size 1024 \
    --n_replays 2048 \
    --n_kernels 96 \
    --vf_coef 1.0