#!/usr/bin/env bash

GAMES_PREFIX='data/games'

mkdir -p ${GAMES_PREFIX}/all
mkdir -p ${GAMES_PREFIX}/train
mkdir -p ${GAMES_PREFIX}/eval

mv ${GAMES_PREFIX}/*.csv ${GAMES_PREFIX}/all/

ls ${GAMES_PREFIX}/all | shuf | head -2048 | while read file
do
mv ${GAMES_PREFIX}/all/${file} ${GAMES_PREFIX}/train/
done

cp ${GAMES_PREFIX}/all/* ${GAMES_PREFIX}/eval/
cp ${GAMES_PREFIX}/train/* ${GAMES_PREFIX}/all/
