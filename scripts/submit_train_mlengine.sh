#!/usr/bin/env bash

python setup.py sdist

#TRAINER_PACKAGE_PATH="/path/to/your/application/sources"
MAIN_TRAINER_MODULE="alphasoccer.actor_critic.run_soccer"
PACKAGE_STAGING_PATH="gs://alphasoccer/package/"

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="training_$now"
JOB_DIR=gs://alphasoccer/jobs/${JOB_NAME}


gcloud ml-engine jobs submit training ${JOB_NAME} \
    --job-dir ${JOB_DIR} \
    --packages dist/alphasoccer-0.1.tar.gz \
    --module-name ${MAIN_TRAINER_MODULE} \
    --region us-central1 \
    --config config.yaml \
    --runtime-version 1.5 \
    -- \
    --n_total_timesteps 1 \
    --n_self_play_games 8 \
    --n_evaluation_games 8 \
    --n_training_steps 32 \
    --batch_size 32 \
    --new_best_model_threshold 0.30 \
    --n_games_in_replay_checkpoint 2 \
    --model_dir ${JOB_DIR}/models/test/model/ \
    --log_dir ${JOB_DIR}/models/test/logs/ \
    --replay_dir ${JOB_DIR}/models/test/replays