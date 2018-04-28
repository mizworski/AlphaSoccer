#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="resnet_standard_size"
LOG_DIR=models/logs/${JOB_NAME}_${now}
MODEL_DIR=models/${JOB_NAME}
REPLAY_DIR=data/replays/${JOB_NAME}

python3 alphasoccer/actor_critic/run_soccer.py \
    --model_dir ${MODEL_DIR} \
    --log_dir ${LOG_DIR} \
    --replay_dir ${REPLAY_DIR} \
    --n_total_timesteps 100 \
    --n_evaluation_games 100 \
    --n_evaluations 10 \
    --n_training_steps 1024 \
    --batch_size 512 \
    --n_games_in_replay_checkpoint 256 \
    --learning_rate 1e-4 \
    --n_rollouts 500 \
    --n_replays 2048 \
    --n_self_play_games 2048 \
    --c_puct 10.0 \
    --n_kernels 128 \
    --residual_blocks 8 \
    --vf_coef 1.5 \
    --moves_before_decaying 3

# set n_replays == n_self_play_games