#! /usr/bin/python3
import time
import os
import configparser
def start_inference(save_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    cfg_scale = float(config.get('config', 'cfg_scale'))
    steps = int(config.get('config', 'steps'))
    cmd = f'python3 \
    /home/linaro/workspace/PixArt-sigma/scripts/interface.py \
    --t5_feat_path {data_path}\
    --save_path {save_path}\
    --cfg_scale {cfg_scale}\
    --steps {steps}\
    --use_t5'
    os.system(cmd)

def start_service():
    last_modified = 0
    while True:
        if os.path.exists(data_path):
            modified_time = os.path.getmtime(data_path)
            if modified_time > last_modified:
                last_modified = modified_time
                start_inference(f'/sd_data/output/vis/{time.strftime("%Y-%m-%d,%H:%M:%S", time.localtime())}.jpg')
        time.sleep(2)

if __name__ == '__main__':
    data_path = '/sd_data/cache/pixart/tokens.npy'
    config_path = '/sd_data/cache/pixart/model_config.ini'
    try:
        os.remove(data_path)
        print(f"Deleted {data_path}")
    except OSError as e:
        print(f"Error deleting {data_path}: {e}")
    start_service()