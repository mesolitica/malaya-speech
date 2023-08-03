import sys
import os

SOURCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
sys.path.insert(0, SOURCE_DIR)

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import base64
import numpy as np
import malaya_speech
from malaya_speech import Pipeline
from malaya_speech.utils.astype import float_to_int
from malaya_speech.streaming import socket
from malaya_speech.streaming import stream
import soundfile as sf

app = FastAPI()


vad_model = malaya_speech.vad.webrtc()
p_vad = Pipeline()
pipeline = (
    p_vad.map(lambda x: float_to_int(x, divide_max_abs=False))
    .map(vad_model)
)

model = malaya_speech.stt.transducer.pt_transformer(
    model='mesolitica/conformer-medium-malay-whisper'
)
_ = model.eval()
p_asr = Pipeline()
pipeline_asr = (
    p_asr.map(lambda x: model.beam_decoder([x])[0], name='speech-to-text')
)

audio = socket.Audio(vad_model=p_vad)


@app.get('/')
async def get():
    with open('index.html') as fopen:
        html = fopen.read()
    return HTMLResponse(html)


@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        """
        RIFF\x1e\xd1\x00\x00WAVEfmt
        RecordRTC send minimum 1508 frame size, WebRTC VAD accept 320, so we have to slice.
        """
        array = np.frombuffer(base64.b64decode(data), dtype=np.int16)[25:]
        array = array.astype(np.float32, order='C') / 32768.0
        frames = list(audio.vad_collector(array))
        try:
            r = stream(
                vad_model=p_vad,
                asr_model=p_asr,
                frames=frames,
                realtime_print=False,
            )
        except BaseException:
            r = []
        print(r)

        if len(r) and len(r[0]['asr_model']):
            sf.write('writing_file_output.wav', r[0]['wav_data'], 16000)
            await websocket.send_text(r[0]['asr_model'])
