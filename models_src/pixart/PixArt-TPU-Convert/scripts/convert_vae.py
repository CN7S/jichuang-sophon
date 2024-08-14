# vae
# 
#



import os
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

from diffusers.models import AutoencoderKL
import torch

import torch.nn as nn
import numpy as np
dtype = torch.float32
device = "cpu"
model_name='/data/jichuang/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers/vae'
onnxpath = "Onnx/VAE/fp32/VAE.onnx"

vae = AutoencoderKL.from_pretrained(model_name).to(device).to(dtype)

class VAE_ONNX(nn.Module):
    def __init__(self,
                 vae):
        super().__init__()
        self.vae = vae
    
    def forward(self, samples):
        samples = self.vae.decode(samples / self.vae.config.scaling_factor).sample
        return samples

vaemodel = VAE_ONNX(vae)
vaemodel.eval()
vaemodel.to(dtype)

# samples = np.load('./samples.npy')
# samples = torch.tensor(samples).to(torch.float32)

inputs = np.load("output/data/input_pixart.npz")
samples = torch.tensor(inputs['samples']).to(device).to(dtype)

np.savez('/home/jichuang/sophon/work/vae/data/input_data.npz', samples = samples.cpu())
input_names = ["samples"]
output_names = ["sample_output"]
print(samples.shape)
torch.onnx.export(vaemodel, 
                  samples,
                  onnxpath, 
                  export_params=True,
                  opset_version=16,
                #   do_constant_folding=True,
                  # keep_initializers_as_inputs = True,
                  input_names=input_names, 
                  output_names=output_names)