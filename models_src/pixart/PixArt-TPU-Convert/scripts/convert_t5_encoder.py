# t5 encoder  
# env : t5 custom env, for custom t5 model
# conda activate t5modver
#

import torch
from transformers import T5EncoderModel, T5Tokenizer

import torch.nn as nn
import numpy as np
from tqdm import tqdm

class Encoder(nn.Module):
    def __init__(self,
                 encoder):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, inputs_embeds, attention_mask):
        # caption_embs = self.encoder.block[0].layer[0].layer_norm(inputs_embeds)
        caption_embs = self.encoder(inputs_embeds, attention_mask=attention_mask)[0]
        return caption_embs

class Block(nn.Module):
    def __init__(self,
                 encoder,
                 layer_id):
        super().__init__()
        self.encoder = encoder
        self.convert_layer = layer_id
    
    def forward(self, inputs_embeds, attention_mask=None, position_bias=None):
        # caption_embs = self.encoder.block[0].layer[0].layer_norm(inputs_embeds)
        output = self.encoder(inputs_embeds, 
                              attention_mask=attention_mask, 
                              position_bias=position_bias, 
                              convert_layer_id=self.convert_layer)
        if self.convert_layer == 0:
            hidden_state=output[0]
            position_bias=output[2]
            return hidden_state, position_bias
        elif self.convert_layer < 24:
            hidden_state=output[0]
            return hidden_state
        else:
            return output[0]

# config
pipeline_load_from = '/data/jichuang/pretrained_models/t5-v1_1-xl'
# pipeline_load_from = '/data/jichuang/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers'
cache_data = True
max_token_length = 300
# caption_token = tokenizer(prompts, max_length=max_sequence_length, padding="max_length", truncation=True,
#                           return_tensors="pt").to(device)


tokenizer = T5Tokenizer.from_pretrained(pipeline_load_from, subfolder="tokenizer")
text_encoder = T5EncoderModel.from_pretrained(pipeline_load_from, 
                                                subfolder="text_encoder",
                                                # load_in_8bit=True,device_map="auto"
                                                ).to("cpu").get_encoder().to(torch.float32)
# T5Encoder = Encoder(text_encoder).eval()


# prepare data
prompts = ['A boy beside the lake.']

input_embeddings = text_encoder.get_input_embeddings()
np.save('/home/jichuang/Onnx/T5-L/fp32/input_embedding_weight.npy', input_embeddings.weight.data)
caption_token = tokenizer(prompts, max_length=max_token_length, padding="max_length", truncation=True,
                        return_tensors="pt")
def padding_caption(caption, length):
    caption.attention_mask[0][:length] = 1
    return caption
caption_token = padding_caption(caption_token, 50)
inputs_embeds = input_embeddings(caption_token.input_ids)
attention_mask = caption_token.attention_mask.to(torch.float32)
position_bias = None

T5Encoder = Encoder(text_encoder).eval()
np.savez('output/data/t5_test_inputs.npz', inputs_embeds = inputs_embeds.detach().numpy(), attention_mask = attention_mask.detach().numpy())
caption_emb = T5Encoder(inputs_embeds, attention_mask)
np.save('output/data/t5_feat_origin.npy', caption_emb.detach().numpy())
# input('stop')
# layer_id = 0
for layer_id in tqdm(range(0,25)):
    
    # get model
    T5Block = Block(text_encoder, layer_id).eval()

    # samples = np.load('./samples.npy')
    # samples = torch.tensor(samples).to(torch.float32)
    if layer_id == 0:
    #first block layer
        input_names = ["inputs_embeds", "attention_mask"]
        output_names = ['hidden_state', 'position_bias']
        model_inputs_tuple = (inputs_embeds, attention_mask)
    elif (layer_id < 24):
    # middle block layer
        # inputs_array = np.load('output/data/t5_layer_output.npz')
        # inputs_embeds = torch.tensor(inputs_array['hidden_state'])
        # position_bias = torch.tensor(inputs_array['position_bias'])
        input_names = ["inputs_embeds", 'attention_mask', 'position_bias']
        output_names = ['hidden_state']
        model_inputs_tuple = (inputs_embeds, attention_mask, position_bias)
    else:
    # final layer
        input_names = ["hidden_state"]
        output_names = ["caption_embs"]
        model_inputs_tuple = (inputs_embeds)

    # convert to Onnx
    model_save_path = f"/home/jichuang/Onnx/T5-L/fp32/T5-block-{layer_id}.onnx" if layer_id < 24 else '/home/jichuang/Onnx/T5-L/fp32/T5-final-layer.onnx'
    torch.onnx.export(T5Block, 
                    model_inputs_tuple,
                    model_save_path, 
                    export_params=True,
                    opset_version=12,
                    # do_constant_folding=True,
                    # keep_initializers_as_inputs = True,
                    input_names=input_names, 
                    output_names=output_names)

    output_save = T5Block(inputs_embeds, attention_mask, position_bias)
    if(layer_id == 0):
        inputs_embeds = output_save[0]
        position_bias = output_save[1]
    elif(layer_id < 24):
        inputs_embeds = output_save
    if(cache_data):
        if(layer_id == 0):
            np.savez(f'output/data/t5_layer_{layer_id}_output.npz', 
                    hidden_state = output_save[0].detach().numpy(),
                    position_bias = output_save[1].detach().numpy())
        elif(layer_id < 24):
            np.savez(f'output/data/t5_layer_{layer_id}_output.npz', 
                    hidden_state = output_save.detach().numpy())
        else:
            np.save('output/data/t5_feat.npy', output_save.detach().numpy())