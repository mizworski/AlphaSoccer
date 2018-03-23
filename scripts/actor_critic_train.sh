#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python3 alphasoccer/actor_critic/run_soccer.py \
    --n_total_timesteps 100 \
    --n_self_play_games 256 \
    --n_evaluation_games 50 \
    --n_evaluations 10 \
    --n_training_steps 512 \
    --batch_size 512 \
    --n_games_in_replay_checkpoint 128 \
    --double_first_self_play
#    --model_dir models/test/model/ \
#    --log_dir models/test/logs/ \
#    --replay_dir models/test/replays \
#    --skip_first_self_play