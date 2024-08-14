from transformers import T5Tokenizer
import numpy as np
import os
class T5Preprocess():
    def __init__(self,
                 load_from : str = 'model/TextFeature',
                 max_sequence_length = 300):
        self.max_sequence_length = max_sequence_length
        self.tokenizer = T5Tokenizer.from_pretrained(load_from, subfolder="tokenizer")
    
    def work(self, prompt : str):
        caption_token = self.tokenizer(prompt, 
                                  max_length=self.max_sequence_length, 
                                  padding="max_length", 
                                  truncation=True,
                                return_tensors="np")
        return caption_token.input_ids

class PixArtClient():
    def __init__(self,
                 ip_addr:str = '192.168.150.1',
                 port= 22,
                 keypath:str = 'ssh_key/shaolin',):
        self.preprocess = T5Preprocess()
        self.ip = ip_addr
        self.port = port
        self.keypath = keypath

    def sendfile(self, filepath, targetpath):
        cmd = f"scp -i {self.keypath} {filepath} linaro@{self.ip}:{targetpath}"
        return os.system(cmd)
    def recvfile(self, filepath, targetpath):
        cmd = f"scp -i {self.keypath} linaro@{self.ip}:{targetpath} {filepath}"
        return os.system(cmd)

    # def set_config(self, ):
    # '/sd_data/image_show/image.jpg'
    def active(self, prompt : str):
        token_path = 'cache/model_input/tokens.npy'
        device_token_path = '/sd_data/cache/pixart/tokens.npy'
        input_ids = self.preprocess.work(prompt)
        np.save(token_path, input_ids)
        self.sendfile(token_path, device_token_path) # to start

