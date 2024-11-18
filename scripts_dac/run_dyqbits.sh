#!/bin/bash

runname=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 10)

CUDA_VISIBLE_DEVICES=1 nohup python dlrm_s_pytorch.py \
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
    --dyqbits-flag \
    --dyqbits-options=32,16,8 \
    --dyqbits-loss-temp=0.03125 \
    --tensor-board-filename=tensorboard/dyqbits/${runname} \
    --save-model=./checkpoints/dyqbits/dlrm > run_kaggle_pt_dyqbits.log 2>&1 &
