#!/bin/bash

runname=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 10)

CUDA_VISIBLE_DEVICES=0 nohup python dlrm_s_pytorch.py \
    --arch-sparse-feature-size=16 \
    --arch-mlp-bot="13-512-256-64-16" \
    --arch-mlp-top="512-256-1" \
    --data-generation=dataset \
    --data-set=kaggle \
    --raw-data-file=./input/train.txt \
    --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
    --loss-function=bce \
    --round-targets=True \
    --learning-rate=0.1 \
    --mini-batch-size=128 \
    --print-freq=800 \
    --print-time \
    --test-freq=8000 \
    --test-mini-batch-size=10240 \
    --test-num-workers=16 \
    --use-gpu \
    --tensor-board-filename=tensorboard/baseline/${runname} \
    --load-model-v2="./checkpoints/baseline/dlrm.epoch0.iter304000" > baseline.pretrained.log 2>&1 &
