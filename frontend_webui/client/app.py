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
from client import Asrllm, StreamingCustomCallbackHandler



class GPTCallbackHandler(StreamingCustomCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        super().on_llm_new_token(token, **kwargs)
        global gpt_output_str
        gpt_output_str = gpt_output_str + token
            


g_status="waiting for input"
g_gpt_text=""

gpt_output_str = ''
gpt_pixart=Asrllm(callbackhandler = GPTCallbackHandler())

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
        return gpt_output_str #text from GPT
    return inner

def show_image(image_path,scale_factor):
    def inner():
        img = Image.open(image_path)
    
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
    config['config']['sampling steps'] = str(steps)  # 将变量 a 写入
    config['config']['cfg scale'] = str(cfg)  # 将变量 b 写入 


    # 将配置写入到指定的 ini 文件中
    with open(file_path, 'w') as configfile:
        config.write(configfile)
    
with gr.Blocks(css=".gradio-container {height:100vh;width:100vw}") as demo:
    
    gr.Markdown("# Text to Image System")
    with gr.Row():
        with gr.Column(scale=0,min_width=600):
            describe = gr.Textbox(label="Describe text")
            text_btn = gr.Button("Send text")
            steps = gr.Slider(0, 20,step=1,label="smapling steps")
            cfg = gr.Slider(0, 10,step=0.5,label="cfg_scale")
            process_btn = gr.Button("Configure parameter")
            record_btn = gr.Button("Sound recording")
            name = gr.Textbox(label="Status bar",value = set_text(),every=1,interactive=False)
            gpt = gr.Textbox(label="GPT",value = set_gpt_text() ,every=1,lines=8)
        with gr.Column(scale=1,min_width=600):
            image = gr.Image(label="Generated image",value = show_image("cache/image/image.jpg",4),every=1)

    @process_btn.click(inputs=[steps,cfg])	
    def process(steps, cfg):
        write_config('cache/config/model_config.ini',cfg,steps) # 将配置写入到指定的 ini 文件中
        global g_status
        if steps ==0 or cfg ==0:
            g_status = "Configure failed!"
        else: g_status = "Configure finish: "+"steps: "+str(steps)+" ,cfg: "+str(cfg)+"!"
        '''while True:
            images.append(Image.open(img_path))
            global images=
            time.sleep(1)  # 程序暂停1秒'''
            
    @text_btn.click(inputs=describe)
    def text(describe):
        with open('/home/jichuang/Desktop/describe.txt','w',encoding='utf-8') as file:
            file.write(describe)
        global g_status
        if describe == "":
            g_status = "Please input describe!"
        else: g_status = "Text input completed: "+describe
            
    #sound recording function
    @record_btn.click()
    def record():
        global gpt_output_str
        gpt_output_str=''
        question = gpt_pixart.asr_work()
        gpt_pixart.llm_work(question)
        
        
        
    


   

demo.launch()

