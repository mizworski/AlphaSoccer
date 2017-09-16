#!/usr/bin/env bash

MODEL_DIR=model
TRAIN_DATA_DIR=data/games/train
EVAL_DATA_DIR=data/games/eval
TRAIN_STEPS=1024

gcloud ml-engine local train --package-path src.models.tensorflow \
                           --module-name src.models.tensorflow.trainer \
                           -- \
                           --model_dir ${MODEL_DIR} \
                           --train_data_dir $TRAIN_DATA_DIR \
                           --eval_data_dir $EVAL_DATA_DIR \
                           --num_epochs $TRAIN_STEPS \
                           --batch_size 256