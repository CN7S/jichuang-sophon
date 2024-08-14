#! /bin/bash
#/data/jichuang/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth
model_path='/data/jichuang/pretrained_models/train/output/PixArt-L-512-0810/checkpoints/epoch_10_step_100286.pth'
python scripts/inference.py \
    --model_path ${model_path} \
    --pipeline_load_from /data/jichuang/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers \
    --image_size 512 \
    --txt_file asset/samples.txt