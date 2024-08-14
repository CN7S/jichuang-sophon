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

from tpu_model import T5_ENCODE_Slice

import numpy as np
import configparser
import time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--last_layer', default=23, type=int) # 0-23 layer
    parser.add_argument('--inputs_file_path', default='/sd_data/models/T5Encoder/input_tokens.npy', type=str) # npy for tokens
    parser.add_argument('--output', default='/sd_data/output/t5_feat_final.npz', type=str)
    parser.add_argument('--t5_path', default='/sd_data/models/T5Encoder', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    do_final_norm = args.last_layer==23
    net = T5_ENCODE_Slice(
        model_path=args.t5_path,
        layer_range=[0, args.last_layer],
        do_final_norm=do_final_norm,
        on_device=False,
        )
    inputs = np.load(args.inputs_file_path)
    attention_mask = np.zeros((1,300)).astype(np.float32)
    attention_mask[:,:50] = 1.0
    if do_final_norm:
        caption_embs = net.forward(inputs, attention_mask)
        np.savez(args.output, caption_embs=caption_embs)
    else:
        [hidden_state, position_bias] = net.forward(inputs, attention_mask)
        np.savez(args.output, hidden_state=hidden_state, position_bias=position_bias)