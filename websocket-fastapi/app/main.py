import sys
import os

DEBUG_SAVE = os.environ.get('DEBUG_SAVE', 'false') == 'true'
MODEL = os.environ.get('MODEL', 'mesolitica/conformer-tiny-ctc')
MODEL_LM = os.environ.get('MODEL_LM', 'mesolitica/kenlm-pseudolabel-whisper-large-v3')
LM_ALPHA = float(os.environ.get('LM_ALPHA', '0.2'))
LM_BETA = float(os.environ.get('LM_BETA', '1.0'))
MIN_LENGTH = float(os.environ.get('MIN_LENGTH', '1.0'))
SILENCE_TIMEOUT = float(os.environ.get('SILENCE_TIMEOUT', '1.0'))
IMPORT_LOCAL = os.environ.get('IMPORT_LOCAL', 'false') == 'true'

if IMPORT_LOCAL:
    SOURCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
    sys.path.insert(0, SOURCE_DIR)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import time
import json
import base64
import numpy as np
import torch
import torchaudio
import math
import malaya_speech
from itertools import groupby
from huggingface_hub import hf_hub_download
from transformers import AutoModel
from pyctcdecode import build_ctcdecoder
import kenlm
from malaya_speech import Pipeline
from malaya_speech.utils.astype import float_to_int
from malaya_speech.streaming import socket
from malaya_speech.streaming import stream
from datetime import datetime
import soundfile as sf

HF_CTC_VOCAB = [
    '',
    'a',
    'b',
    'c',
    'd',
    'e',
    'f',
    'g',
    'h',
    'i',
    'j',
    'k',
    'l',
    'm',
    'n',
    'o',
    'p',
    'q',
    'r',
    's',
    't',
    'u',
    'v',
    'w',
    'x',
    'y',
    'z',
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    ' ',
    '?',
    '_'
]
HF_CTC_VOCAB_INDEX = {no: c for no, c in enumerate(HF_CTC_VOCAB)}
HF_CTC_VOCAB_REV = {v: k for k, v in HF_CTC_VOCAB_INDEX.items()}

DECIBEL = 2 * 20 * math.log10(torch.iinfo(torch.int16).max)
GAIN = pow(10, 0.05 * DECIBEL)

spectrogram_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=400, n_mels=80, hop_length=160)


def piecewise_linear_log(x):
    x = x * GAIN
    x[x > math.e] = torch.log(x[x > math.e])
    x[x <= math.e] = x[x <= math.e] / math.e
    return x


def melspectrogram(x):
    if isinstance(x, np.ndarray):
        x = torch.Tensor(x)
    x = spectrogram_transform(x).transpose(1, 0)
    return piecewise_linear_log(x)


app = FastAPI()
SR = 16000

vad_model = malaya_speech.vad.webrtc()
p_vad = Pipeline()
pipeline = (
    p_vad.map(lambda x: float_to_int(x, divide_max_abs=False))
    .map(vad_model)
)

lm = hf_hub_download(MODEL_LM, 'out.binary')
kenlm_model = kenlm.Model(lm)
decoder = build_ctcdecoder(
    HF_CTC_VOCAB,
    kenlm_model,
    alpha=LM_ALPHA,
    beta=LM_BETA,
    ctc_token_idx=len(HF_CTC_VOCAB) - 1
)

model = AutoModel.from_pretrained(MODEL, trust_remote_code=True)
mel = melspectrogram(np.zeros((SR * 5,)))
inputs = {
    'inputs': mel.unsqueeze(0),
    'lengths': torch.tensor([len(mel)])
}
model(**inputs)

p_asr = Pipeline()

index = 0


def predict(y):
    global index
    if DEBUG_SAVE:
        sf.write(f'{index}.mp3', y, SR)
    before = time.time()
    mel = melspectrogram(y)
    inputs = {
        'inputs': mel.unsqueeze(0),
        'lengths': torch.tensor([len(mel)])
    }
    r = model(**inputs)
    logits = r[0].detach().numpy()
    argmax = np.argmax(logits, axis=-1)
    tokens = ''.join([HF_CTC_VOCAB_INDEX[k] for k in argmax[0]])
    grouped_tokens = [token_group[0] for token_group in groupby(tokens)]
    filtered_tokens = list(filter(lambda token: token != '_', grouped_tokens))
    r = ''.join(filtered_tokens).strip()
    if len(r) < 2:
        return
    out = decoder.decode_beams(logits[0], prune_history=True)
    d_lm, lm_state, timesteps, logit_score, lm_score = out[0]
    time_taken = time.time() - before
    index += 1
    d = {
        'predict': r,
        'predict_lm': d_lm,
        'time_taken': time_taken
    }
    return json.dumps(d)


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
                    t_ = t_['speech-to-text']
                    if t_ is not None:
                        text = t_

                    manager.wav_data[client_id] = np.array([], dtype=np.float32)
                    manager.length[client_id] = 0

            if len(text):
                await manager.send_personal_message(text, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket, client_id=client_id)
