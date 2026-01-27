#!/bin/bash

# good: resnet18; mobilenet_v2; mobilenet_v3_small
# medium: efficientnet_b0; resnet34
MODEL_ARCH="resnet18"

BATCH_SIZE=16		# default: 16
LEARNING_RATE=0.001	# default: 0.1
EPOCHS=150		    # default: 30

DATA_DIR="../data/"
OUTPUT_DIR="../models/${MODEL_ARCH}"

python trainCBI.py \
    "$DATA_DIR" \
    --model-dir="$OUTPUT_DIR" \
    --batch-size=$BATCH_SIZE \
    --learning-rate=$LEARNING_RATE \
    --epochs=$EPOCHS \
    --arch=$MODEL_ARCH \
    --gpu=0 \
    --pretrained
