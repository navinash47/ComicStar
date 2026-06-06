#!/bin/bash

mkdir -p /workspace/ComfyUI/models/checkpoints
mkdir -p /workspace/ComfyUI/models/vae
mkdir -p /workspace/ComfyUI/models/loras
mkdir -p /workspace/ComfyUI/models/controlnet
mkdir -p /workspace/ComfyUI/models/ipadapter
mkdir -p /workspace/ComfyUI/models/clip_vision

cd /workspace/ComfyUI/models/checkpoints

wget -O illustrious-xl-v0.1.safetensors \
"https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/resolve/main/Illustrious-XL-v0.1.safetensors"

cd /workspace/ComfyUI/models/vae

wget -O sdxl-vae-fp16-fix.safetensors \
"https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors"
