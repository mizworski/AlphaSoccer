#!/usr/bin/env bash

JOB_NAME="test_$(date +%Y%m%d_%H%M%S)"

BUCKET_NAME="cnn-test"
REGION=europe-west1
JOB_NAME="custom_test_$(date +%Y%m%d_%H%M%S)"

OUTPUT_PATH=gs://$BUCKET_NAME/jobs/${JOB_NAME}

MODEL_DIR=${OUTPUT_PATH}/model
TRAIN_FILES=gs://cnn-test/data/links/train.csv
EVAL_FILES=gs://cnn-test/data/links/eval.csv
TRAIN_STEPS=1024

gcloud ml-engine jobs submit training $JOB_NAME \
                                    --runtime-version 1.2 \
                                    --job-dir $OUTPUT_PATH \
                                    --region $REGION \
                                    --package-path trainer/ \
                                    --module-name trainer.soccer_train \
                                    --config config.yaml \
                                    -- \
                                    --model_dir ${MODEL_DIR} \
                                    --train_files $TRAIN_FILES \
                                    --eval_files $EVAL_FILES \
                                    --num_epochs $TRAIN_STEPS \
                                    --batch_size 256