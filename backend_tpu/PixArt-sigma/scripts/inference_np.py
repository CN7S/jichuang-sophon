#! /usr/bin/python3
import os
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import warnings
warnings.filterwarnings("ignore")  # ignore warning
import re
import argparse
from datetime import datetime
from tools.image import save_image

from diffusion.model.utils import prepare_prompt_ar
from diffusion import DPMS

from tpu_model import PixArtMS, VAE_DECODE, T5_ENCODE_Slice

import numpy as np
import configparser
import time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--version', default='sigma', type=str)
    parser.add_argument('--task_config_path', default='/home/linaro/workspace/PixArt-sigma/config/config.ini', type=str)
    parser.add_argument('--t5_path', default='/sd_data/models/T5Encoder', type=str)
    parser.add_argument('--pixartpath', default='/sd_data/models/PixArtMS', type=str)
    parser.add_argument('--vaepath', default='/sd_data/models/VAE_PixArt', type=str)
    parser.add_argument('--sampling_algo', default='dpm-solver', type=str, choices=['iddpm', 'dpm-solver', 'sa-solver'])
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--clear', default=1, type=int)

    return parser.parse_args()


def set_env(seed=0):
    np.random.seed(seed)
    for _ in range(30):
        np.random.randn(1, 4, args.image_size, args.image_size)

def visualize(datapath, use_t5, sample_steps, cfg_scale, save_path):

    latent_size_h, latent_size_w = latent_size, latent_size
    if use_t5: # datapath is a npy file for tokens
        # call t5 inference
        cmd = f'python3 /home/linaro/workspace/PixArt-sigma/scripts/t5_inference.py \
        --last_layer {last_layer} \
        --t5_path {args.t5_path} \
        --inputs_file_path {datapath} \
        --output {t5_feat_middle_path}'
        print(cmd)
        os.system(cmd)
        feat_data = np.load(t5_feat_middle_path)
        # finish the last compute
        caption_embs = text_encoer.forward(feat_data['hidden_state'], feat_data['position_bias'])
    else: # datapath is a npz file for embedding data
        inputs = np.load(datapath)
        caption_embs = inputs['caption_embs']
    if(len(caption_embs.shape) == 3):
        caption_embs = caption_embs[:,None]
    print(f'finish embedding')

    # Create sampling noise:
    n = 1 #len(prompts)
    z = np.random.randn(n, 4, latent_size_h, latent_size_w)
    model_kwargs = dict(mask=emb_masks)
    dpm_solver = DPMS(model.forward_with_dpmsolver,
                        condition=caption_embs,
                        uncondition=null_y,
                        cfg_scale=cfg_scale,
                        model_kwargs=model_kwargs)

    # sample_start_time = time.time()

    samples = dpm_solver.sample(
        z,
        steps=sample_steps,
        order=2,
        skip_type="time_uniform",
        method="multistep",
    )

    # sample_end_time = time.time()
    # sample_duration = sample_end_time - sample_start_time

    # print(f"sample total time : {sample_duration}s, steps = {sample_steps}")

    samples = vae.forward(samples)

    # vae_end_time = time.time()
    # vae_duration = vae_end_time - sample_end_time 
    # print(f"vae total time : {vae_duration}s")



    # Save images:
    os.umask(0o000)  # file permission: 666; dir permission: 777
    for i, sample in enumerate(samples):
        print("Saving path: ", save_path)
        save_image(sample, save_path, nrow = 1, normalize=True, value_range=(-1, 1))
        save_image(sample, show_path, nrow = 1, normalize=True, value_range=(-1, 1))
        # os.system(f'python3 image.py --image_path {save_path}')        
def start_service():
    last_modified = 0
    while True:
        if os.path.exists(args.task_config_path):
            modified_time = os.path.getmtime(args.task_config_path)
            if modified_time > last_modified:
                print('')
                last_modified = modified_time
                config = configparser.ConfigParser()
                config.read(args.task_config_path)
                t5_feat_path = config.get('config','t5_feat_path')
                use_t5 = config.getboolean('config','use_t5')
                save_path = config.get('config','save_path')
                cfg_scale = float(config.get('config','cfg_scale'))
                sample_steps = int(config.get('config','steps'))
                visualize(t5_feat_path, use_t5, sample_steps, cfg_scale, save_path)
        time.sleep(1)
        sys.stdout.write('\rfree to work')
        sys.stdout.flush()

if __name__ == '__main__':
    args = get_args()
    show_path = '/sd_data/image_show/image.jpg'
    if(args.clear):
        try:
            os.remove(args.task_config_path)
            print(f"Deleted {args.task_config_path}")
        except OSError as e:
            print(f"Error deleting {args.task_config_path}: {e}")
    

    
    seed = args.seed
    set_env(seed)

    assert args.sampling_algo in ['iddpm', 'dpm-solver', 'sa-solver']

    # only support fixed latent size currently
    latent_size = args.image_size // 8
    max_sequence_length = 50 # max_sequence_length = {"alpha": 120, "sigma": 300}[args.version]
    pe_interpolation = args.image_size / 512
    micro_condition = False
    weight_dtype = np.float32
    emb_masks = np.load('/sd_data/dataset/const_emb.npy')
    null_y = np.load(os.path.join(args.t5_path, 't5_null_feat.npy'))
    # null_y = np.load('/sd_data/dataset/t5_null_feat.npy')

    #t5 config
    last_layer = 2
    t5_feat_middle_path = '/sd_data/cache/t5/feat_data.npz'

    print(f"Inference with {weight_dtype}")

    # model setting / bmodel load

    model = PixArtMS(
            model_path = args.pixartpath
    )

    vae = VAE_DECODE(
            model_path = args.vaepath
    )

    text_encoer = T5_ENCODE_Slice(
        model_path=args.t5_path,
        layer_range=[last_layer+1, 23],
        do_final_norm=True,
        on_device=True,
        )

    start_service()
