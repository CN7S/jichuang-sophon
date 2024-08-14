import os
import sys
import argparse
from datetime import datetime
import numpy as np
import configparser
import time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default = '/home/linaro/workspace/PixArt-sigma/config/config.ini', type=str)
    parser.add_argument('--t5_feat_path', default = '/sd_data/dataset/t5_feat_test.npz', type=str)
    parser.add_argument('--save_path', default = '/sd_data/output/vis/test.jpg', type=str)
    parser.add_argument('--cfg_scale', default=4.5, type=float)
    parser.add_argument('--steps', default=20, type=int)
    parser.add_argument('--use_t5', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    # Create A Config Object
    config = configparser.ConfigParser()

    # Config 
    config['config'] = {
        't5_feat_path': args.t5_feat_path,
        'save_path': args.save_path,
        'use_t5': f'{args.use_t5}',
        'cfg_scale': f'{args.cfg_scale}',
        'steps': f'{args.steps}'
    }

    # Write File
    with open(f"{args.config_path}.tmp", 'w') as configfile:
        config.write(configfile)
    
    import shutil
    shutil.move(f"{args.config_path}.tmp", args.config_path)

    print('successfully generate config.')
