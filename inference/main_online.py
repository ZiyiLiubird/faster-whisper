import os, sys
from fastapi import FastAPI
from typing import Optional, List, Union
import base64
import time
from time import time as ttime
from scipy.io.wavfile import read, write
from pydantic import BaseModel
import json


from faster_whisper import WhisperModel

model_size = "large-v3"
model_path = "/data1/ziyiliu/checkpoints/Systran/faster-whisper-large-v3"
# Run on GPU with FP16
model = WhisperModel(model_path, device="cuda", compute_type="float16")


class TTSResponse(BaseModel):  # 定义一个类用作返回值
    #现在没有使用，因为audio太大会导致转pydantic速度太慢
    audio: str 
    sampling_rate: int

class TTSRequest(BaseModel):  # 定义一个类用作参数
    text: str
    top_k: int
    temperature: float


app = FastAPI()

@app.get("/")
async def read_root():
    return {"name": "ASR-Serving"}


@app.post("/tts_torch")
async def tts_torch(param: TTSRequest):
    #print("DEBUG tts_torch post",type(param))
    start_time = time.time()
    text = param.text
    temperature = param.temperature
    top_k = param.top_k
    top_p = 1

    print("INIT FILE TIME",time.time()-start_time)
    # res =  torch_engine.tts_fn(text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, language, reference_audio, emotion, prompt_mode, style_text, style_weight)

    sample_rate, audio_output = torch_engine.get_tts_wav(text=text, text_language=text_language,
                                                         top_k=top_k,
                                                         top_p=top_p,
                                                         temperature=temperature)

    print("TTS_FN TORCH INFER",time.time()-start_time)
    
    # res_str = base64.b64encode(res[1][1].tostring())
    res_str = base64.b64encode(audio_output.tostring())
    response = TTSResponse(audio=res_str, sampling_rate=sample_rate)
    #print("DEBUG TTSResponse")
    print("TTS TORCH cost time ",time.time()-start_time)
    print("*"*100)
    return response



