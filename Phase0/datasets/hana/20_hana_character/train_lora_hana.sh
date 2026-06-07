#!/usr/bin/env bash

# Training script placeholder for LoRA (hana_v1)
# Edit the variables below before running.

DATA_DIR="/workspace/datasets/hana/20_hana_character"
OUTPUT_DIR="/workspace/ComfyUI/models/loras"
OUTPUT_NAME="hana_v1"
SD_CHECKPOINT="<path-to-stable-diffusion-checkpoint-or-hub-model>"

mkdir -p "$OUTPUT_DIR"

# Example command using sd-scripts style train_network.py and accelerate.
# Replace the placeholder variables as needed and install dependencies first.

accelerate launch --num_cpu_threads_per_process 4 train_network.py \
  --pretrained_model_name_or_path "$SD_CHECKPOINT" \
  --train_data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --save_model_as safetensors \
  --network_module lora \
  --network_dim 128 \
  --learning_rate 1e-4 \
  --max_train_steps 1000 \
  --train_batch_size 1 \
  --mixed_precision fp16 \
  --save_precision fp16 \
  --output_name "$OUTPUT_NAME"

# After successful training the file will be at:
# $OUTPUT_DIR/${OUTPUT_NAME}.safetensors

# Notes:
# - This script assumes you have `train_network.py` available (from sd-scripts / training repo) and
#   `accelerate` configured for your GPU environment.
# - Adjust --max_train_steps, batch size, learning rate, and network_dim to taste.
