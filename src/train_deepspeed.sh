#!/bin/bash

# Activate your virtual environment if you have one
# source venv/bin/activate

# --- Configuration --- #
NUM_GPUS_PER_NODE=3 # Adjust based on your node configuration
DS_CONFIG="ds_config.json" # Path to DeepSpeed config file
MASTER_PORT=$(shuf -i 10000-65535 -n 1) # Assign a random free port

# --- Base Arguments (Common to all models) --- #
BASE_ARGS="--task_type classification --labeling_strategy directional \
 --epochs 10 --early_stopping_patience 30 \
 --num_workers 0 --seed 42 \
 --lr 0.001 --deepspeed --deepspeed_config $DS_CONFIG"

# --- Model Specific Arguments --- #

# MODEL_ARGS="--model_name transformer --d_model 512 --nhead 8 --num_layers 2 --dim_ff 256 --dropout 0.1"
# MODEL_ARGS="--model_name lstm --hidden_dim 512 --num_layers 2 --dropout 0.1"
MODEL_ARGS="--model_name tcn --hidden_dim 512 --num_layers 2 --dropout 0.1"

# --- Construct the full command --- #
# Note: Batch size, LR, weight decay are typically set in ds_config.json and handled by DeepSpeed based on the global batch size.
# If you need to override them from the command line, you might need custom handling in your script
# or ensure your ds_config.json is set up to accept command-line overrides if supported.
CMD="deepspeed --num_gpus=$NUM_GPUS_PER_NODE --master_port $MASTER_PORT src/train_deepspeed.py $BASE_ARGS $MODEL_ARGS"

# Print the command being executed
echo "Executing: $CMD"

# Execute the command
$CMD 


# bash src/train_deepspeed.sh