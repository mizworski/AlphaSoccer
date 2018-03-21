#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python3 alphasoccer/actor_critic/run_soccer.py
