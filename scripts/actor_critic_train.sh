#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python3 src/actor_critic/run_soccer.py
