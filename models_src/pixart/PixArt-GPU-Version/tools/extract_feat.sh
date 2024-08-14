#! /bin/bash 
dataset_name=comic-dataset-small
json_file=/data/jichuang/dataset/${dataset_name}/InternData/data_info.json
dataset_root=/data/jichuang/dataset/${dataset_name}/InternImgs/

python tools/extract_features.py \
    --run_t5_feature_extract \
    --run_vae_feature_extract \
    --multi_scale \
    --vae_json_file $json_file \
    --vae_model /data/jichuang/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers/vae/ \
    --dataset_root  $dataset_root \
    --vae_save_root /data/jichuang/dataset/${dataset_name}/InternData/ \
    --max_length 40 \
    --t5_json_path $json_file \
    --t5_models_dir /data/jichuang/pretrained_models/t5-v1_1-xl/ \
    --t5_save_root /data/jichuang/dataset/${dataset_name}/InternData/

