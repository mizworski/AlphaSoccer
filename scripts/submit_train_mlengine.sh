#!/usr/bin/env bash

python setup.py sdist

#TRAINER_PACKAGE_PATH="/path/to/your/application/sources"
MAIN_TRAINER_MODULE="alphasoccer.actor_critic.run_soccer"
PACKAGE_STAGING_PATH="gs://alphasoccer/package/"

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="training_$now"
JOB_DIR=gs://alphasoccer/jobs/$JOB_NAME


gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR \
    --packages dist/alphasoccer-0.1.tar.gz \
    --module-name $MAIN_TRAINER_MODULE \
    --region us-central1 \
    --config config.yaml \
    --runtime-version 1.5