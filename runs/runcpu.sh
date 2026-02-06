#!/bin/bash

# Showing an example run for exercising some of the code paths on the CPU (or MPS on Macbooks)
# This script was last updated/tuned on Jan 17, 2026.

# Run as:
# bash runs/runcpu.sh

# NOTE: Training LLMs requires GPU compute and $$$. You will not get far on your Macbook.
# Think of this run as educational/fun demo, not something you should expect to work well.
# You may also want to run this script manually and one by one, copy pasting commands into your terminal.

# See the train.sh script for more details, configuration options and comments.

export EXTRA_DEPENDENCIES=cpu
export WRITE_REPORT=0
export DATASET_SHARDS_TO_DOWNLOAD=8
export NPROC_PER_NODE=1
# I tuned this run to complete in about 30 minutes on my MacBook Pro M3 Max.
# To get better results, try increasing num_iterations, or get other ideas from your favorite LLM.
export MODEL_DEPTH=6
export DEVICE_BATCH_SIZE=32
export EVAL_BATCH_SIZE=1
export EXTRA_BASE_TRAINING_ARGS="--head-dim=64 --window-pattern=L --max-seq-len=512 --total-batch-size=16384 --eval-every=100 --eval-tokens=524288 --core-metric-every=-1 --sample-every=100 --num-iterations=5000"
export EXTRA_BASE_EVAL_ARGS="--split-tokens=16384 --max-per-task=16"
export EXTRA_SFT_TRAINING_ARGS="--max-seq-len=512 --total-batch-size=16384 --eval-every=200 --eval-tokens=524288 --num-iterations=1500"
export RUN_SFT_EVAL=0
bash $(dirname $(realpath $0))/train.sh

# Chat with the model over CLI
# The model should be able to say that it is Paris.
# It might even know that the color of the sky is blue.
# Sometimes the model likes it if you first say Hi before you ask it questions.
# python -m scripts.chat_cli -p "What is the capital of France?"

# Chat with the model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web
