#! /bin/bash

dataset_name=dataset-comic
pretrained_models=/data/jichuang/pretrained_models/train/output/PixArt-512-New/checkpoints/epoch_10_step_22481.pth
#/data/jichuang/pretrained_models/train/output/PixArt-512-New/checkpoints/epoch_10_step_22481.pth
#/data/jichuang/pretrained_models/train/output/PixArt-512-0804/checkpoints/epoch_40_step_101761.pth 
#/data/jichuang/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth

python -m torch.distributed.launch \
    --nproc_per_node=1 --master_port=12345 \
               train_scripts/train.py \
                configs/pixart_sigma_config/PixArt_sigma_l2_img512_internalms.py \
              --load-from ${pretrained_models}   \
                --work-dir  /data/jichuang/pretrained_models/train/output/PixArt-L-512-0813   \
        --pipeline_load_from /data/jichuang/pretrained_models/t5-v1_1-xl/
        #/data/jichuang/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers
        #/data/jichuang/pretrained_models/t5-v1_1-xl/