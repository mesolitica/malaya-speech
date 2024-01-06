import sys
import os

DEBUG_SAVE = os.environ.get('DEBUG_SAVE', 'false') == 'true'
MODEL = os.environ.get('MODEL', 'mesolitica/malaysian-whisper-base')
FORCE_LANGUAGE = os.environ.get('FORCE_LANGUAGE', 'ms')
IMPORT_LOCAL = os.environ.get('IMPORT_LOCAL', 'false') == 'true'
USE_CTRANSLATE2 = os.environ.get('USE_CTRANSLATE2', 'false') == 'true'
CTRANSLATE2_OUTPUT = os.environ.get('CTRANSLATE2_OUTPUT', 'out')
MIN_LENGTH = float(os.environ.get('MIN_LENGTH', '2.0'))
SILENCE_TIMEOUT = float(os.environ.get('SILENCE_TIMEOUT', '1.0'))

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
from datetime import datetime
import soundfile as sf
import ctranslate2
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

app = FastAPI()


vad_model = malaya_speech.vad.webrtc()
p_vad = Pipeline()
pipeline = (
    p_vad.map(lambda x: float_to_int(x, divide_max_abs=False))
    .map(vad_model)
)

processor = AutoProcessor.from_pretrained(MODEL)

if USE_CTRANSLATE2:
    if not os.path.exists(os.path.join(CTRANSLATE2_OUTPUT, 'model.bin')):
        converter = ctranslate2.converters.TransformersConverter(MODEL)
        converter.convert(CTRANSLATE2_OUTPUT, quantization='int8', force=True)
    model = ctranslate2.models.Whisper(CTRANSLATE2_OUTPUT)
else:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL)
    _ = model.eval()

p_asr = Pipeline()

REJECTED_TEXT = {
    'Terima kasih kerana menonton!',
    'Terima kasih.'
}

index = 1


def predict(y):
    global index
    if DEBUG_SAVE:
        sf.write(f'{index}.mp3', y, 16000)
    if USE_CTRANSLATE2:
        inputs = processor(y, return_tensors='np', sampling_rate=16000)
        features = ctranslate2.StorageView.from_array(inputs.input_features)
        prompt = processor.tokenizer.convert_tokens_to_ids(
            [
                '<|startoftranscript|>',
                f'<|{FORCE_LANGUAGE}|>',
                '<|transcribe|>',
            ]
        )
        results = model.generate(features, [prompt])
        transcription = processor.decode(results[0].sequences_ids[0])
    else:
        inputs = processor([y], return_tensors='pt', sampling_rate=16000)
        r = model.generate(
            inputs['input_features'],
            language=FORCE_LANGUAGE,
            return_timestamps=True)
        transcription = processor.tokenizer.decode(r[0], skip_special_tokens=True)
    index += 1
    return transcription


pipeline_asr = (
    p_asr.map(lambda x: predict(x), name='speech-to-text')
)


sample_rate = 16000


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.audio = {}
        self.queue = {}
        self.wav_data = {}
        self.length = {}
        self.silent = {}

    async def connect(self, websocket: WebSocket, client_id):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.audio[client_id] = socket.Audio(
            vad_model=p_vad,
            enable_silent_timeout=True,
            silent_timeout=SILENCE_TIMEOUT
        )
        self.queue[client_id] = np.array([], dtype=np.float32)
        self.wav_data[client_id] = np.array([], dtype=np.float32)
        self.length[client_id] = 0
        self.silent[client_id] = datetime.now()

    def disconnect(self, websocket: WebSocket, client_id):
        self.active_connections.remove(websocket)
        self.audio.pop(client_id, None)
        self.queue.pop(client_id, None)
        self.wav_data.pop(client_id, None)
        self.length.pop(client_id, None)
        self.silent.pop(client_id, None)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)


manager = ConnectionManager()


@app.get('/')
async def get():
    f = 'index.html'
    if IMPORT_LOCAL:
        f = os.path.join('./app', f)
    with open(f) as fopen:
        if USE_CTRANSLATE2:
            model_name = f'Ctranslate2 {MODEL}'
        else:
            model_name = MODEL
        model_name = f'{model_name} forcing language {FORCE_LANGUAGE}'
        html = fopen.read().replace('{{model}}', model_name)
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
                    manager.silent[client_id] = datetime.now()

                if frame is None and (
                        manager.length[client_id] >= MIN_LENGTH or
                        (datetime.now() - manager.silent[client_id]).seconds >= SILENCE_TIMEOUT
                ):
                    if (
                        not len(manager.wav_data[client_id]) or
                        np.mean(manager.wav_data[client_id]) == 0
                    ):
                        continue

                    wav_data = np.concatenate(
                        [np.zeros(shape=(int(0.05 * sample_rate),)), manager.wav_data[client_id]])

                    t_ = p_asr(wav_data)
                    text = t_['speech-to-text']

                    manager.wav_data[client_id] = np.array([], dtype=np.float32)
                    manager.length[client_id] = 0

            if len(text):
                if text.strip() in REJECTED_TEXT:
                    text = ''
                else:
                    await manager.send_personal_message(text, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket, client_id=client_id)
