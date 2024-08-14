from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

from typing import Optional, Any
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live

from .pixart_preprocess import PixArtClient

llm: Optional[LlamaCpp] = None
callback_manager: Any = None
template_tiny = """<|system|>
                   You are a smart mini computer named Raspberry Pi. 
                   Write a short but funny answer.</s>
                   <|user|>
                   {question}</s>
                   <|assistant|>"""

template_file = """<|system|>
                   You are a translator. 
                   Write a prompt from the question, and only use a short sentence.
                   The prompt should only include your description of a picture, so that another AI can generate a picture from it.</s>
                   <|user|>
                   {question}</s>
                   <|assistant|>"""

template_file = """[INST] <<SYS>>
                   You are a story teller.
                   Before telling the story, write a prompt , and only use a short sentence starts with <pr> and ends with </pr>.
                   The prompt should only include your description of a picture, so that another AI can generate a picture from it.
                   Then tell a story related to the question. </SYS>>
                    {question} [/INST]"""

template_llama = """<s>[INST] <<SYS>>
                    You are a smart mini computer named Raspberry Pi.
                    Write a short but funny answer.</SYS>>
                    {question} [/INST]"""

class StreamingCustomCallbackHandler(StreamingStdOutCallbackHandler):
    def __init__(self):
        super().__init__()
        self.prompt_ = ''
        self.tpu_client = PixArtClient()
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        super().on_llm_new_token(token, **kwargs)
        # print(f"{token}", end="")
        self.prompt_ = self.prompt_ + token
        position = self.prompt_.find("</pr>",-10)
        if(position != -1):
            begin_ptr = self.prompt_.find("<pr>") + 4
            end_ptr = position
            prompt = self.prompt_[begin_ptr:end_ptr]
            self.tpu_client.active(prompt)
            self.prompt_ = ''

class Asrllm():
    def __init__(self,
                callbackhandler=StreamingCustomCallbackHandler(),
                asr_model_path = "model/whisper",
                template = template_file,
                llm_model_path = "model/llama-2-7b/llama-2-7b-chat.Q2_K.gguf",
      ):
        self.asr_model_id = asr_model_path
        self.template = template
        self.model_file = llm_model_path
        self.transcriber = pipeline("automatic-speech-recognition",
                       model=self.asr_model_id,
                       device="cpu")
        self.llm_init(callbackhandler)
    
    def  transcribe_mic ( self, chunk_length_s: float ) -> str :  
        sampling_rate = 16000
        Sample_rate = self.transcriber.feature_extractor.sampling_rate 
    
        mic = ffmpeg_microphone_live( 
                sampling_rate=sampling_rate, 
                chunk_length_s=chunk_length_s, 
                stream_chunk_s=chunk_length_s, 
            ) 
        
        result = "" 
        for item in self.transcriber(mic): 
            result = item[ "text" ] 
            if not item[ "partial" ][ 0 ]: 
                break 
        return result.strip()
    
    def llm_init(self, callbackhandler = StreamingCustomCallbackHandler()):
        """ Load large language model """
        self.callback_manager = CallbackManager([callbackhandler])
        self.llm = LlamaCpp(
            model_path=self.model_file,
            temperature=0.1,
            n_gpu_layers=0,
            n_batch=256,
            callback_manager=self.callback_manager,
            verbose=True,
        )
    
    
    def llm_start(self, question: str):
        """ Ask LLM a question """
        self.prompt = PromptTemplate(template=self.template, input_variables=["question"])
        self.chain = self.prompt | self.llm | StrOutputParser()
        self.chain.invoke({"question": question}, config={})
    
    def asr_work(self):
        question = self.transcribe_mic(chunk_length_s=5.0) 
        if len(question) > 0:
            print(f">{question}")
            print("")
        return question
        
    def llm_work(self, question):
        if len(question) > 0:
            self.llm_start(question)
        
if __name__ == "__main__":

    asrllm = Asrllm()
    while True:
        print("Start Speaking")
        asrllm.asr_work()
        asrllm.llm_work()

    
    
    
