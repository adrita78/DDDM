#!/bin/bash

# Project root — CHANGE THIS
PROJ=/path/to/DDDM-main

cd "$PROJ" || exit 1

MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --dropout 0.3"

# VE + Pseudo-LPIPS + c = 0.000069
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear --VP False --use_pl True --c 0.000069"

TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --epochs 1000"

# Persistent logging/checkpoints
export DDDM_LOGDIR="$PROJ/checkpoints"
export DIFFUSION_BLOB_LOGDIR="$PROJ/checkpoints/VE_PL_c000069"

mkdir -p "$DDDM_LOGDIR"
mkdir -p "$DIFFUSION_BLOB_LOGDIR"

# Verify dataset exists
if [ ! -d "$PROJ/datasets/cifar_train" ]; then
    echo "ERROR: CIFAR-10 dataset not found at $PROJ/datasets/cifar_train"
    exit 1
fi

echo "========================================"
echo "VE + Pseudo-LPIPS training"
echo "c = 0.000069"
echo "========================================"

torchrun \
    --standalone \
    --nproc_per_node=2 \
    "$PROJ/scripts/image_train.py" \
    --data_dir "$PROJ/datasets/cifar_train" \
    $MODEL_FLAGS \
    $DIFFUSION_FLAGS \
    $TRAIN_FLAGS


