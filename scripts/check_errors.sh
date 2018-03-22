#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

mkdir -p models/test/
rm -r models/test/*

python3 alphasoccer/actor_critic/run_soccer.py \
        --n_self_play_games 8 \
        --n_evaluation_games 8 \
        --n_training_steps 32 \
        --batch_size 32 \
        --new_best_model_threshold 0.30 \
        --n_games_in_replay_checkpoint 2 \
        --double_first_self_play False \
        --model_dir models/test/model/ \
        --log_dir models/test/logs/ \
        --replay_dir models/test/replays
