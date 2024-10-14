    


import requests
import base64
from pydub import AudioSegment
import numpy
import time
from scipy.io.wavfile import read, write
import base64
import numpy as np
import json
import librosa


if __name__ == "__main__":

    # url="http://wa-tts.parametrix.cn/"
    url="http://0.0.0.0:63061/"

    input_file = "leidian.mp3"
    # audio = AudioSegment.from_mp3(input_file)
    # binary_data = audio.raw_data
    y, sr = librosa.load(input_file)
    audio = y.tolist()
    params = {
        "audio": audio,
    }
    
    start_time = time.time()


    start_time2 = time.time()
    response = requests.post(url + "tts_torch", json=params)
    # print(response.json())
    response_dict = response.json()
    text = response_dict['text']
    print(text)
    ed = time.time()
    print("cost time", ed - start_time)
