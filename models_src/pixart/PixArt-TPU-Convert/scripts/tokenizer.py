import argparse
from transformers import T5EncoderModel, T5Tokenizer
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline_load_from", default='/data/jichuang/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers',
        type=str, help="Download for loading text_encoder, "
                       "tokenizer and vae from https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
    )

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    prompts = 'A boy play with a dog.'

    tokenizer = T5Tokenizer.from_pretrained(args.pipeline_load_from, subfolder="tokenizer")
    
    caption_token = tokenizer(prompts, max_length=300, padding="max_length", truncation=True,
                            return_tensors="np")
    
    np.save('output/tokens.npy', caption_token.input_ids)
