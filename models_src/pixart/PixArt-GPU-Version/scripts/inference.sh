#! /bin/bash
# "/data/jichuang/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth"
model="/data/jichuang/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth"
#/data/jichuang/pretrained_models/train/output/PixArt-L-512-0810/checkpoints/epoch_10_step_100286.pth
#"/data/jichuang/pretrained_models/PixArt_512_CUS.pth"
# "/data/jichuang/pretrained_models/train/output/PixArt-512-0804/checkpoints/epoch_10_step_50936.pth"
# "/data/jichuang/pretrained_models/train/output/PixArt-512-0804/checkpoints/epoch_40_step_101761.pth"
# "/data/jichuang/pretrained_models/train/output/PixArt-512-0802/checkpoints/epoch_10_step_23911.pth"
python scripts/inference.py \
    --image_size 512 \
    --pipeline_load_from /data/jichuang/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers \
    --model_path $model \
    --t5_path /data/jichuang/pretrained_models/t5-v1_1-xl/ \
    --txt_file /home/jichuang/PixArt-sigma_L/asset/prompt-1.txt 

