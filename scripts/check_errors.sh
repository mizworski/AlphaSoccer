#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

#mkdir -p models/test/
#rm -r models/test/logs
#rm -r models/test/*

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="training_$now"

python3 alphasoccer/actor_critic/run_soccer.py \
    --n_total_timesteps 1 \
    --n_self_play_games 4 \
    --n_evaluation_games 4 \
    --n_evaluations 2 \
    --n_training_steps 64 \
    --batch_size 32 \
    --new_best_model_threshold 0.10 \
    --n_games_in_replay_checkpoint 2 \
    --model_dir models/test/model/ \
    --log_dir models/test/logs/${JOB_NAME} \
    --replay_dir models/test/replays \
    --skip_first_self_play