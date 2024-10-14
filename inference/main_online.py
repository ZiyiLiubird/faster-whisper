import os, sys
from fastapi import FastAPI
from typing import Optional, List, Union
import base64
import time
from time import time as ttime
from scipy.io.wavfile import read, write
from pydantic import BaseModel
import json
import numpy as np

from faster_whisper import WhisperModel

model_size = "large-v3"
model_path = "/data1/ziyiliu/checkpoints/Systran/faster-whisper-large-v3"
# Run on GPU with FP16
model = WhisperModel(model_path, device="cuda", compute_type="float16")


class ASRResponse(BaseModel):  # 定义一个类用作返回值
    #现在没有使用，因为audio太大会导致转pydantic速度太慢
    text: str 

class ASRRequest(BaseModel):  # 定义一个类用作参数
    audio: list


app = FastAPI()

@app.get("/")
async def read_root():
    return {"name": "ASR-Serving"}


@app.post("/tts_torch")
async def tts_torch(param: ASRRequest):
    #print("DEBUG tts_torch post",type(param))
    start_time = time.time()
    audio = param.audio
    audio = np.array(audio)
    
    print("INIT FILE TIME",time.time()-start_time)
    # res =  torch_engine.tts_fn(text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, language, reference_audio, emotion, prompt_mode, style_text, style_weight)
    segments, info = model.transcribe(audio, language="en", beam_size=5)
    print("ASR_FN TORCH INFER",time.time()-start_time)
    segments = list(segments)
    texts = [seg.text for seg in segments]
    text = '.'.join(texts)

    print("trans",time.time()-start_time)
    # print("ASR_FN TORCH INFER",time.time()-start_time)
    response = ASRResponse(text=text)
    # res_str = base64.b64encode(res[1][1].tostring())
    return response



