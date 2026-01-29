#!/bin/bash

# good: resnet18; mobilenet_v2; mobilenet_v3_small
# medium: efficientnet_b0; resnet34

# Configuration for multiple models
# Format: "model_name|batch_size|learning_rate|epochs"
MODELS=(
    "mobilenet_v2|16|0.01|60"
    "mobilenet_v2|32|0.001|20"
    "mobilenet_v2|16|0.001|40"
    "resnet18|32|0.01|50"
    "resnet18|32|0.001|50"
)

SEED=42	
DATA_DIR="../data/"

# Train each model with its specific configuration
for model_config in "${MODELS[@]}"; do
    IFS='|' read -r MODEL_ARCH BATCH_SIZE LEARNING_RATE EPOCHS <<< "$model_config"
    
    OUTPUT_DIR="../models/${MODEL_ARCH}_b${BATCH_SIZE}_lr${LEARNING_RATE}_e${EPOCHS}"
    
    echo "=========================================="
    echo "Training $MODEL_ARCH"
    echo "  Batch Size: $BATCH_SIZE"
    echo "  Learning Rate: $LEARNING_RATE"
    echo "  Epochs: $EPOCHS"
    echo "  Output: $OUTPUT_DIR"
    echo "=========================================="
    
    python trainCBI.py \
        "$DATA_DIR" \
        --model-dir="$OUTPUT_DIR" \
        --batch-size=$BATCH_SIZE \
        --learning-rate=$LEARNING_RATE \
        --epochs=$EPOCHS \
        --arch=$MODEL_ARCH \
        --seed=$SEED \
        --gpu=0 \
        --pretrained
    
    if [ $? -eq 0 ]; then
        echo "✓ Training completed for $MODEL_ARCH"
    else
        echo "✗ Training failed for $MODEL_ARCH"
    fi
    echo ""
done

echo "=========================================="
echo "All training jobs completed!"
echo "=========================================="
