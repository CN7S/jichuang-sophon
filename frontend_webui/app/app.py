#! /usr/bin/python
import os
import sys
from pathlib import Path
current_path = Path(__file__).resolve()
sys.path.insert(0, str(current_path.parent.parent))

from datetime import datetime
import gradio as gr
import configparser


from PIL import Image
from typing import Any
from client import Asrllm, StreamingCustomCallbackHandler, PixArtClient



class GPTCallbackHandler(StreamingCustomCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        super().on_llm_new_token(token, **kwargs)
        global gpt_output_str
        gpt_output_str = gpt_output_str + token
        #now_position = gpt_output_str.find("<pr>", -10)
        #if(now_position != -1):
        #    self.flag = 0
        #elif self.flag == 0:
        #    now_position = gpt_output_str.find("<\pr>", -10)
        #    if(now_position != -1):
        #        self.flag = 1
        #if(self.flag == 1):
        #    now_position = gpt_output_str.find('.' , self.last_position + 1)
        #    if(now_position != -1):
        #        cmd = f"echo {got_output_str[self.last_position + 1:now_position]} >> cache/audio/audio.txt"
        #        os.system(cmd)
        #    self.last_position = now_position
        
            

pixartClient = PixArtClient()
remote_img_path = '/sd_data/image_show/image.jpg'
remote_cfg_path = '/sd_data/cache/pixart/model_config.ini'
voice_file_path = 'cache/audio/audio.txt'
g_status="waiting for input"
describe_text=''
gpt_output_str = ''
gpt_str_tokens = []
gpt_end_ptr = 0
gpt_pixart=Asrllm(callbackhandler = GPTCallbackHandler())

def set_describe_text():
    def inner():
        return describe_text
    return inner

def set_text():
    def inner():
        return g_status
    return inner
    
#GPT text print function
def set_gpt_text():
    def inner():
        # now = datetime.now()
        # current_second = now.strftime("-%s")
        # global g_gpt_text
        # g_gpt_text += current_second
        global gpt_end_ptr
        
        # get tokens
        if len(gpt_output_str) > gpt_end_ptr:
            ptr1 = gpt_output_str[gpt_end_ptr:].find('.')
            ptr2 = gpt_output_str[gpt_end_ptr:].find('?')
            ptr3 = gpt_output_str[gpt_end_ptr:].find('!')
            ptr4 = gpt_output_str[gpt_end_ptr:].find(':')
            gpt_end_ptr_tmp = -1
            if ptr1 != -1:
                gpt_end_ptr_tmp = ptr1+gpt_end_ptr+1
            elif ptr2 != -1:
                gpt_end_ptr_tmp = ptr2+gpt_end_ptr+1
            elif ptr3 != -1:
                gpt_end_ptr_tmp = ptr3+gpt_end_ptr+1
            elif ptr4 != -1:
                gpt_end_ptr_tmp = ptr4+gpt_end_ptr+1
            
                
            # print(ptr1,ptr2,ptr3, gpt_end_ptr, gpt_end_ptr_tmp)
            if(gpt_end_ptr_tmp != -1):
                tmp_str = gpt_output_str[gpt_end_ptr:gpt_end_ptr_tmp]
                gpt_end_ptr = gpt_end_ptr_tmp
                cmd = f'echo \"{tmp_str}\" >> {voice_file_path}'
                os.system(cmd)
            
            #new_tokens = gpt_output_str[gpt_end_ptr:gpt_end_ptr_tmp].split()
            #for token in new_tokens:
            #    gpt_str_tokens.append(token)
            #gpt_end_ptr = gpt_output_str.find(gpt_str_tokens[-1], max(-len(gpt_output_str), -len(gpt_str_tokens[-1]) - 5))
            #print('gpt_end_ptr:', gpt_end_ptr)
            #print('tokens:', gpt_str_tokens)
            #if len(gpt_output_str) == 1:
            #    gpt_str_tokens = []
            #else:
            #    gpt_str_tokens = gpt_str_tokens[:-1]
        
        # pop tokens
        #if():
        #    tmp_str = ''
        #    for i in range(8):
        #        tmp_str = tmp_str + ' ' + gpt_str_tokens[i]
        #   cmd = f'echo \"{tmp_str}\" >> {voice_file_path}'
        #    os.system(cmd)
        #    cmd = f'echo \"{tmp_str}\"'
        #    # os.system(cmd)
        #    gpt_str_tokens = gpt_str_tokens[8:]
        
        devide = len(gpt_output_str)//100
        if devide==0:
            show_gpt_text = gpt_output_str
        else: show_gpt_text = gpt_output_str[devide * 100 : len(gpt_output_str)]
        
        # voice
        
        
        
        #
        
        
        return gpt_output_str # show_gpt_text #text from GPT
    return inner

def show_image(image_path,scale_factor):
    def inner():
        pixartClient.recvfile(image_path, remote_img_path)
        if os.path.exists(image_path):
            img = Image.open(image_path)
        else :
            img = Imgae.open('cache/image/icon.jpg')
        original_width,original_height = img.size
    
        new_width = int (original_width *scale_factor)
        new_height = int (original_height * scale_factor)
    
        resize_img = img.resize((new_width,new_height),Image.LANCZOS)
    
        return resize_img
    return inner

def write_config(file_path,cfg,steps):
    # 创建一个 ConfigParser 对象
    config = configparser.ConfigParser()

    # 添加节（section）和键值对（key-value pairs）
    config['config'] = {}  # 创建默认节
    config['config']['steps'] = str(steps)  # 将变量 a 写入
    config['config']['cfg_scale'] = str(cfg)  # 将变量 b 写入 --


    # 将配置写入到指定的 ini 文件中
    with open(file_path, 'w') as configfile:
        config.write(configfile)
    pixartClient.sendfile(file_path, remote_cfg_path)


with gr.Blocks() as demo:
    
    #gr.Markdown("# Text to Image System")
    with gr.Row():
        with gr.Column(scale=0,min_width=375): 
            describe = gr.Textbox(label="Describe text", value=set_describe_text(), every = 1)
            with gr.Row():
                # trecord_btn = gr.Button("Voice input",size="sm")
                text_re_btn = gr.Button("Record text",size="sm")
                text_btn = gr.Button("Send text",size="sm")
            with gr.Row():
                steps = gr.Slider(5, 20,step=1, value = 10, label="smapling steps")
                cfg = gr.Slider(0, 10,step=0.5, value = 4.5,label="cfg_scale")
            process_btn = gr.Button("Configure parameter",size="sm")
            name = gr.Textbox(label="Status bar",value = set_text(),every=1,interactive=False)
            record_btn = gr.Button("Sound recording",size="sm")
            gpt = gr.Textbox(label="GPT",value = set_gpt_text() ,every=1,lines=2,max_lines=2,elem_id="gpt_textbox")
        with gr.Column(scale=0,min_width=550):
            image = gr.Image(label="Generated image",value = show_image("cache/image/image.jpg",2),every=1)

    @process_btn.click(inputs=[steps,cfg])	
    def process(steps, cfg):
        write_config('cache/config/model_config.ini',cfg,steps) # 将配置写入到指定的 ini 文件中
        global g_status
        if steps ==0 or cfg ==0:
            g_status = "Configure failed!"
        else: g_status = "Configure finish: "+"steps: "+str(steps)+" ,cfg: "+str(cfg)+"!"

    #send text  
    @text_btn.click(inputs=describe)
    def text(describe):
        # with open('/home/jichuang/Desktop/describe.txt','w',encoding='utf-8') as file:
        #     file.write(describe)
        global g_status
        if describe == "":
            g_status = "Please input describe!"
        else: g_status = "Text input completed: "+describe
        pixartClient.active(describe)
            
    #sound recording--ask to gpt
    @record_btn.click()
    def record():
        global gpt_output_str, gpt_end_ptr
        gpt_output_str=''
        gpt_end_ptr = 0
        question = gpt_pixart.asr_work()
        gpt_pixart.llm_work(question)
    
    @text_re_btn.click()
    def record2():
        global describe_text
        describe_text = gpt_pixart.asr_work()


demo.launch()

