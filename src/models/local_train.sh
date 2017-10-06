#!/usr/bin/env bash


PREFIX=/home/mizworski/PycharmProjects/PaperSoccerRL
MODEL_DIR=${PREFIX}/model
TRAIN_DATA_DIR=${PREFIX}/data/games/train
EVAL_DATA_DIR=${PREFIX}/data/games/eval
N_EPOCHS=16
TRAIN_FILES=${PREFIX}/data/tfrecords/train
EVAL_FILES=${PREFIX}/data/tfrecords/eval
BATCH_SIZE=2048

export PYTHONPATH=${PYTHONPATH}:${PREFIX}

gcloud ml-engine local train --package-path supervised_policy \
                           --module-name supervised_policy.soccer_train \
                           -- \
                           --model_dir ${MODEL_DIR} \
                           --train_data_dir ${TRAIN_DATA_DIR} \
                           --eval_data_dir ${EVAL_DATA_DIR} \
                           --train_files=${TRAIN_FILES} \
                           --eval_files=${EVAL_FILES}\
                           --num_epochs ${N_EPOCHS} \
                           --batch_size ${BATCH_SIZE}