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

from tpu_model import T5_ENCODE, T5_ENCODE_Slice

import numpy as np
import configparser
import time



if __name__ == '__main__':
    # # T5_ENCODE DEBUG
    # net = T5_ENCODE('/sd_data/models/T5Encoder')
    # inputs = np.load('/sd_data/models/T5Encoder/t5_test_inputs.npz')
    # attention_mask = inputs['attention_mask']
    # attention_mask[:,:50] = 1.0
    # caption_embs = net.forward(inputs['inputs_embeds'], inputs['attention_mask'])
    # np.savez('/sd_data/output/t5_feat.npz', caption_embs=caption_embs, attention_mask=attention_mask)

    # T5_SLICE DEBUG
    net = T5_ENCODE_Slice(
        model_path='/sd_data/models/T5Encoder',
        layer_range=[0,18],
        do_final_norm=False,
        on_device=False,
        )
    inputs = np.load('/sd_data/models/T5Encoder/t5_test_inputs.npz')
    attention_mask = np.zeros((1,300)).astype(np.float32)
    attention_mask[:,:50] = 1.0
    [hidden_state, position_bias] = net.forward(inputs['inputs_embeds'], attention_mask)
    net = T5_ENCODE_Slice(
        model_path='/sd_data/models/T5Encoder',
        layer_range=[19,23],
        do_final_norm=True,
        on_device=True,
    )
    [caption_embs] = net.forward(hidden_state, position_bias)
    np.savez('/sd_data/output/t5_feat.npz', caption_embs=caption_embs)

    
