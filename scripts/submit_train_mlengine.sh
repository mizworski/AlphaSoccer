#!/usr/bin/env bash

python setup.py sdist

#TRAINER_PACKAGE_PATH="/path/to/your/application/sources"
MAIN_TRAINER_MODULE="alphasoccer.actor_critic.run_soccer"
PACKAGE_STAGING_PATH="gs://alphasoccer/package/"

now=$(date +"%Y%m%d_%H%M%S")
base=training_20180325_130527
JOB_NAME=training_${now}_base_${base}


JOB_DIR=gs://alphasoccer/jobs/${JOB_NAME}
LOG_DIR=gs://alphasoccer/logs/${JOB_NAME}
MODEL_DIR=gs://alphasoccer/models/${JOB_NAME}
REPLAY_DIR=gs://alphasoccer/replays/${JOB_NAME}

gsutil -m cp gs://alphasoccer/models/${base}/* ${MODEL_DIR}/
gsutil -m cp gs://alphasoccer/replays/${base}/* ${REPLAY_DIR}/

gcloud ml-engine jobs submit training ${JOB_NAME} \
    --job-dir ${JOB_DIR} \
    --packages dist/alphasoccer-0.1.tar.gz \
    --module-name ${MAIN_TRAINER_MODULE} \
    --region us-central1 \
    --config config.yaml \
    --runtime-version 1.5 \
    -- \
    --n_total_timesteps 100 \
    --n_evaluation_games 50 \
    --n_evaluations 10 \
    --n_training_steps 1024 \
    --batch_size 512 \
    --n_games_in_replay_checkpoint 256 \
    --model_dir ${MODEL_DIR} \
    --log_dir ${LOG_DIR} \
    --replay_dir ${REPLAY_DIR} \
    --learning_rate 2e-3 \
    --n_rollouts 750 \
    --n_replays 768 \
    --n_self_play_games 512 \
    --c_puct 5.0