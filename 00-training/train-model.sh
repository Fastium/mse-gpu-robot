#!/bin/bash

# Configuration
PYTHON_CMD="../.venv/bin/python"
DATA_DIR="../data/"
GPU_ID=0
SEED=42  # fixed seed for reproducibility

# ==============================================================================
# PARAMETERS GRID
# Define the values you want to test for each parameter.
# The script will run EVERY combination of these lists.
# ==============================================================================

# List of architectures
ARCHITECTURES=(
    # "mobilenet_v3_small"
    "resnet18"
    # "efficientnet_b0"
    "mobilenet_v2"
    # "resnet34"
)

# List of batch sizes
BATCH_SIZES=(
    16
    32
    # 64
)

# List of learning rates
LEARNING_RATES=(
    # 0.1
    0.01
    0.001
    # 0.0001
)

# List of epochs
EPOCHS_LIST=(
    30
    40
    50
    60
)

# ==============================================================================
# SCRIPT LOGIC
# ==============================================================================

# Calculate total experiments
total_exps=$(( ${#ARCHITECTURES[@]} * ${#BATCH_SIZES[@]} * ${#LEARNING_RATES[@]} * ${#EPOCHS_LIST[@]} ))
current_exp=0

echo "=================================================================="
echo "Starting Grid Search"
echo "Total experiments planned: $total_exps"
echo "Python Command: $PYTHON_CMD"
echo "=================================================================="

# Nested loops to generate every combination
for ARCH in "${ARCHITECTURES[@]}"; do
    for BATCH in "${BATCH_SIZES[@]}"; do
        for LR in "${LEARNING_RATES[@]}"; do
            for EPOCHS in "${EPOCHS_LIST[@]}"; do

                ((current_exp++))

                # Generate a unique output directory name
                OUTPUT_DIR="../models/${ARCH}_b${BATCH}_lr${LR}_e${EPOCHS}"

                echo ""
                echo "------------------------------------------------------------------"
                echo "Experiment $current_exp / $total_exps"
                echo "Params: Arch=$ARCH, Batch=$BATCH, LR=$LR, Epochs=$EPOCHS"
                echo "Output: $OUTPUT_DIR"
                echo "------------------------------------------------------------------"

                mkdir -p "$OUTPUT_DIR"

                $PYTHON_CMD trainCBI.py \
                    "$DATA_DIR" \
                    --model-dir="$OUTPUT_DIR" \
                    --batch-size=$BATCH \
                    --learning-rate=$LR \
                    --epochs=$EPOCHS \
                    --arch=$ARCH \
                    --gpu=$GPU_ID \
                    --seed=$SEED \
                    --workers=0 \
                    --pretrained

                if [ $? -eq 0 ]; then
                    echo "✅ Success"
                else
                    echo "❌ Failure"
                fi

                # Pause between runs
                sleep 2

            done
        done
    done
done

echo ""
echo "=================================================================="
echo "Grid Search Completed."
echo "=================================================================="
