import sys
import os

DEBUG_SAVE = os.environ.get('DEBUG_SAVE', 'false') == 'true'
MODEL = os.environ.get('DEBUG_SAVE', 'mesolitica/malaysian-whisper-base')
IMPORT_LOCAL = os.environ.get('IMPORT_LOCAL', 'false') == 'true'

if IMPORT_LOCAL:
    SOURCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
    sys.path.insert(0, SOURCE_DIR)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import base64
import numpy as np
import malaya_speech
from malaya_speech import Pipeline
from malaya_speech.utils.astype import float_to_int
from malaya_speech.streaming import socket
from malaya_speech.streaming import stream
import soundfile as sf
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

app = FastAPI()


vad_model = malaya_speech.vad.webrtc()
p_vad = Pipeline()
pipeline = (
    p_vad.map(lambda x: float_to_int(x, divide_max_abs=False))
    .map(vad_model)
)

processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL)
_ = model.eval()
p_asr = Pipeline()

index = 0


def predict(y):
    global index
    if DEBUG_SAVE:
        sf.write(f'{index}.mp3', y, 16000)
    inputs = processor([y], return_tensors='pt', sampling_rate=16000)
    r = model.generate(inputs['input_features'], language='ms', return_timestamps=True)
    index += 1
    return processor.tokenizer.decode(r[0], skip_special_tokens=True)


pipeline_asr = (
    p_asr.map(lambda x: predict(x), name='speech-to-text')
)


sample_rate = 16000
min_length = 2.0


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.audio = {}
        self.queue = {}
        self.wav_data = {}
        self.length = {}

    async def connect(self, websocket: WebSocket, client_id):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.audio[client_id] = socket.Audio(vad_model=p_vad)
        self.queue[client_id] = np.array([], dtype=np.float32)
        self.wav_data[client_id] = np.array([], dtype=np.float32)
        self.length[client_id] = 0

    def disconnect(self, websocket: WebSocket, client_id):
        self.active_connections.remove(websocket)
        self.queue.pop(client_id, None)
        self.wav_data.pop(client_id, None)
        self.audio.pop(client_id, None)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)


manager = ConnectionManager()


@app.get('/')
async def get():
    f = 'index.html'
    if IMPORT_LOCAL:
        f = os.path.join('./app', f)
    with open(f) as fopen:
        html = fopen.read().replace('{{model}}', MODEL)
    return HTMLResponse(html)


@app.websocket('/ws/{client_id}')
async def websocket_endpoint(websocket: WebSocket, client_id: int):

    await manager.connect(websocket, client_id=client_id)
    try:
        while True:
            data = await websocket.receive_text()
            """
            RIFF\x1e\xd1\x00\x00WAVEfmt
            RecordRTC send minimum 1508 frame size, WebRTC VAD accept 320, so we have to slice.
            """
            array = np.frombuffer(base64.b64decode(data), dtype=np.int16)[24:]

            array = array.astype(np.float32, order='C') / 32768.0
            frames = list(manager.audio[client_id].vad_collector(array))

            text = ''
            for frame in frames:
                if frame is not None:
                    frame, index = frame
                    manager.wav_data[client_id] = np.concatenate(
                        [manager.wav_data[client_id], frame])
                    manager.length[client_id] += frame.shape[0] / sample_rate
                if frame is None and manager.length[client_id] >= min_length:
                    wav_data = np.concatenate(
                        [np.zeros(shape=(int(0.05 * sample_rate),)), manager.wav_data[client_id]])
                    t_ = p_asr(wav_data)
                    text = t_['speech-to-text']
                    manager.wav_data[client_id] = np.array([], dtype=np.float32)
                    manager.length[client_id] = 0

            if len(text):
                await manager.send_personal_message(text, websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket, client_id=client_id)
