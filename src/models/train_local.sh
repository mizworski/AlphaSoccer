#!/usr/bin/env bash
PREFIX=/home/mizworski/PycharmProjects/PaperSoccerRL
MODEL_DIR=${PREFIX}/model
TRAIN_DATA_DIR=${PREFIX}/data/games/train
EVAL_DATA_DIR=${PREFIX}/data/games/eval
TRAIN_STEPS=1024
TRAIN_FILES=${PREFIX}/data/tfrecords/train
EVAL_FILES=${PREFIX}/data/tfrecords/eval

gcloud ml-engine local train --package-path trainer \
                           --module-name trainer.soccer_train \
                           -- \
                           --model_dir ${MODEL_DIR} \
                           --train_data_dir $TRAIN_DATA_DIR \
                           --eval_data_dir $EVAL_DATA_DIR \
                           --train_files=$TRAIN_FILES \
                           --eval_files=$EVAL_FILES\
                           --num_epochs $TRAIN_STEPS \
                           --batch_size 256