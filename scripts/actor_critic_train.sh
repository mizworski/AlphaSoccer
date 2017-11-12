#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python3 src2/actor_critic/trainer.py
