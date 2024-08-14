# # pixart 
# # embedding layer
# # 
# #



# import os
# import sys
# from pathlib import Path
# current_file_path = Path(__file__).resolve()
# sys.path.insert(0, str(current_file_path.parent.parent))

# import torch
# from diffusion.model.nets import PixArtMS_XL_2, PixArt_XL_2, PixArtMS_L_2

# import torch.nn as nn
# import numpy as np
# dtype = torch.float32
# device = "cpu"
# image_size = 512
# latent_size = image_size // 8
# max_sequence_length = 300
# pe_interpolation = image_size / 512
# micro_condition = False
# model_name='/data/jichuang/pretrained_models/train/output/PixArt-L-512-0810/checkpoints/epoch_10_step_100286.pth'
# # '/data/jichuang/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth'
# onnxpath = "/home/jichuang/Onnx/Pixart/fp32/Pixart_512_MS_Embed.onnx"

# model = PixArtMS_L_2(
#     input_size=latent_size,
#     pe_interpolation=pe_interpolation,
#     micro_condition=micro_condition,
#     model_max_length=max_sequence_length,
# ).to(device)
# print("Generating sample from ckpt: %s" % model_name)
# state_dict = torch.load(model_name, map_location=lambda storage, loc: storage)
# if 'pos_embed' in state_dict['state_dict']:
#     del state_dict['state_dict']['pos_embed']
# missing, unexpected = model.load_state_dict(state_dict['state_dict'], strict=False)
# print('Missing keys: ', missing)
# print('Unexpected keys', unexpected)


# # for name, param in model.named_parameters():
# #     print(f"Layer: {name} | Size: {param.size()} | Values : ")

# # input()

# class PinArt_CUSTOM(nn.Module):
#     def __init__(self,
#                  pixart):
#         super().__init__()
#         self.pixart = pixart
    
#     def forward(self, x, timestep, y):
#         ret = self.pixart(x, timestep, y, data_info=None, mask=None)
#         return ret

# pixartmodel = PinArt_CUSTOM(model)
# pixartmodel.eval()
# pixartmodel.to(dtype)

# # samples = np.load('./samples.npy')
# # samples = torch.tensor(samples).to(torch.float32)

# inputs = np.load("output/data/input_pixart.npz")
# x = torch.tensor(inputs['x']).to(device).to(dtype)
# y = torch.tensor(inputs['y']).to(device).to(dtype)
# t = torch.tensor(inputs['t']).to(device).to(dtype)

# print(x.shape, t.shape, y.shape)
# np.savez('/home/jichuang/sophon/work/pixart/data/embedder_input.npz', x = x.detach().numpy(), t = t.detach().numpy(), y = y.detach().numpy())
# input_names = ["x", "t", "y"]
# output_names = ["output_x","output_t","output_y"]
# # output = pixartmodel(x,t,y)
# # print(output)
# # np.save('/home/jichuang/sophon/work/pixart/data/embedder_output.npz', output['x'].detach().numpy())


# # weight = input_embeddings.weight.data.to("cpu").numpy()
# # np.save('tpumodels/input_embeddings.npy', weight)


# # print('weight', weight.shape)
# torch.onnx.export(pixartmodel, 
#                   (x, t, y),
#                   onnxpath, 
#                   export_params=True,
#                   opset_version=16,
#                   # do_constant_folding=True,
#                   # keep_initializers_as_inputs = True,
#                   input_names=input_names, 
#                   output_names=output_names)




# pixart 
# block ver
# 
#



import os
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

import torch
from diffusion.model.nets import PixArtMS_XL_2, PixArt_XL_2, PixArtMS_L_2

import torch.nn as nn
import numpy as np
dtype = torch.float32
device = "cpu"
image_size = 512
latent_size = image_size // 8
max_sequence_length = 300
pe_interpolation = image_size / 512
micro_condition = False
model_name='/data/jichuang/pretrained_models/train/output/PixArt-L-512-0810/checkpoints/epoch_10_step_100286.pth'
#'/data/jichuang/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth'
onnxpath = "/home/jichuang/Onnx/Pixart/fp32/Pixart_512_MS_block.onnx"

model = PixArtMS_L_2(
    input_size=latent_size,
    pe_interpolation=pe_interpolation,
    micro_condition=micro_condition,
    model_max_length=max_sequence_length,
).to(device)
print("Generating sample from ckpt: %s" % model_name)
state_dict = torch.load(model_name, map_location=lambda storage, loc: storage)
if 'pos_embed' in state_dict['state_dict']:
    del state_dict['state_dict']['pos_embed']
missing, unexpected = model.load_state_dict(state_dict['state_dict'], strict=False)
print('Missing keys: ', missing)
print('Unexpected keys', unexpected)


# for name, param in model.named_parameters():
#     print(f"Layer: {name} | Size: {param.size()} | Values : ")

# input()

class PinArt_CUSTOM(nn.Module):
    def __init__(self,
                 pixart):
        super().__init__()
        self.pixart = pixart
    
    def forward(self, x, timestep, y, mask=None):
        ret = self.pixart(x, timestep, y, data_info=None, mask=mask)
        return ret

pixartmodel = PinArt_CUSTOM(model)
pixartmodel.eval()
pixartmodel.to(dtype)

# samples = np.load('./samples.npy')
# samples = torch.tensor(samples).to(torch.float32)

# inputs = np.load("output/data/input_pixart.npz")
# x = torch.tensor(inputs['x']).to(device).to(dtype)
# y = torch.tensor(inputs['y']).to(device).to(dtype)
# t = torch.tensor(inputs['t']).to(device).to(dtype)
# mask = torch.tensor(inputs['mask']).to(device).to(dtype)

# inputs = np.load("output/data/t_block_input.npz")
# t = torch.tensor(inputs['t']).to(device).to(dtype)

inputs = np.load("output/data/output_embedding.npz")
x = torch.tensor(inputs['x']).to(device).to(dtype)
y = torch.tensor(inputs['y']).to(device).to(dtype)
t = torch.tensor(inputs['t']).to(device).to(dtype)
mask = torch.tensor(inputs['mask']).to(device).to(dtype)
print(x.shape, y.shape, t.shape, mask.shape)

if mask is not None:
    if mask.shape[0] != y.shape[0]:
        mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
    mask = mask.squeeze(1).squeeze(1)
    y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
    y_lens = mask.sum(dim=1).tolist()
else:
    y_lens = [y.shape[2]] * y.shape[0]
    y = y.squeeze(1).view(1, -1, x.shape[-1])

input_names = ["x", "t", "y"]
output_names = ["output_x"]
output = pixartmodel(x,t,y)
print(x.shape, y.shape, t.shape, mask.shape)

torch.onnx.export(pixartmodel, 
                  (x, t, y),
                  onnxpath, 
                  export_params=True,
                  opset_version=16,
                  # do_constant_folding=True,
                  # keep_initializers_as_inputs = True,
                  input_names=input_names, 
                  output_names=output_names)










