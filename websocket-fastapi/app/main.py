import sys
import os

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


sample_rate = 16000
min_length = 0.2


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
    with open('./app/index.html') as fopen:
        html = fopen.read()
    return HTMLResponse(html)


@app.websocket("/ws/{client_id}")
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
