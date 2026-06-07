Dataset: hana_character (20 stills)

Contents:
- 001.png .. 020.png — placeholder images for each prompt
- 001.txt .. 020.txt — caption files matching each image

Captions follow the user's spec: the trigger word `hana_character` appears in every caption and emotional/full-body variants are labeled accordingly.

Training script: `train_lora_hana.sh` contains a prepared `accelerate` command (with placeholders) to train a LoRA and save it to `/workspace/ComfyUI/models/loras/hana_v1.safetensors`.

Usage:
1. Replace placeholder images with actual generated images (ensure filenames match `001.png`..`020.png`).
2. Edit `train_lora_hana.sh` to set the path to your SD checkpoint and desired training hyperparameters.
3. Run:

```bash
bash train_lora_hana.sh
```

If you want, I can run the training here — confirm and ensure the environment has GPU, SD checkpoint, and dependencies installed.