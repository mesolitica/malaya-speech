"""
vllm serve "mesolitica/Malaysian-Qwen2.5-7B-Speech-Instruct" --hf_overrides '{"architectures": ["LLMAudioForConditionalGeneration"]}' --trust-remote-code --port 8001
"""

import gradio as gr
import time
import base64
import numpy as np
import requests
import traceback
import io
import os
import librosa
import tempfile
import json
import malaya_speech
import soundfile as sf
from dataclasses import dataclass, field
from threading import Thread
from pydub import AudioSegment
from pydub.generators import Sine
from vad import get_speech_timestamps, collect_chunks, VadOptions
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
elevenlabs = ElevenLabs(
    api_key=ELEVENLABS_API_KEY,
)

ALIVE = """
<svg xmlns="http://www.w3.org/2000/svg" width="16" height="17" viewBox="0 0 16 17" fill="none" class="shrink-0 w-6 h-6 md:w-10 md:h-10 text-public-correct">
<path fill-rule="evenodd" clip-rule="evenodd" d="M8.26868 15C10.1252 15 11.9057 14.2625 13.2184 12.9497C14.5312 11.637 15.2687 9.85652 15.2687 8C15.2687 6.14348 14.5312 4.36301 13.2184 3.05025C11.9057 1.7375 10.1252 1 8.26868 1C6.41216 1 4.63168 1.7375 3.31893 3.05025C2.00617 4.36301 1.26868 6.14348 1.26868 8C1.26868 9.85652 2.00617 11.637 3.31893 12.9497C4.63168 14.2625 6.41216 15 8.26868 15ZM12.1127 6.209C12.2344 6.05146 12.2886 5.85202 12.2633 5.65454C12.2379 5.45706 12.1352 5.27773 11.9777 5.156C11.8201 5.03427 11.6207 4.9801 11.4232 5.00542C11.2257 5.03073 11.0464 5.13346 10.9247 5.291L7.22468 10.081L5.57568 8.248C5.51042 8.17247 5.43075 8.11073 5.34132 8.06639C5.2519 8.02205 5.15452 7.99601 5.0549 7.98978C4.95528 7.98356 4.85542 7.99729 4.76117 8.03016C4.66693 8.06303 4.58019 8.11438 4.50604 8.1812C4.4319 8.24803 4.37184 8.32898 4.32938 8.41931C4.28693 8.50965 4.26293 8.60755 4.2588 8.70728C4.25467 8.807 4.27048 8.90656 4.30532 9.00009C4.34016 9.09363 4.39331 9.17927 4.46168 9.252L6.71168 11.752C6.78517 11.8335 6.87565 11.8979 6.97674 11.9406C7.07782 11.9833 7.18706 12.0034 7.29673 11.9993C7.40639 11.9952 7.51383 11.967 7.61144 11.9169C7.70906 11.8667 7.79448 11.7958 7.86168 11.709L12.1127 6.209Z" fill="currentColor"></path>
</svg>
"""

NOT_ALIVE = """
<svg xmlns="http://www.w3.org/2000/svg" width="16" height="17" viewBox="0 0 16 17" fill="none" role="img" aria-labelledby="atgdzvkoo32chmrozji0gnd8n2boqs6x"><desc id="atgdzvkoo32chmrozji0gnd8n2boqs6x">down icon</desc>
<path fill-rule="evenodd" clip-rule="evenodd" d="M8.26868 15C10.1252 15 11.9057 14.2625 13.2184 12.9497C14.5312 11.637 15.2687 9.85652 15.2687 8C15.2687 6.14348 14.5312 4.36301 13.2184 3.05025C11.9057 1.7375 10.1252 1 8.26868 1C6.41216 1 4.63168 1.7375 3.31893 3.05025C2.00617 4.36301 1.26868 6.14348 1.26868 8C1.26868 9.85652 2.00617 11.637 3.31893 12.9497C4.63168 14.2625 6.41216 15 8.26868 15ZM8.26868 4C8.46759 4 8.65835 4.07902 8.79901 4.21967C8.93966 4.36032 9.01868 4.55109 9.01868 4.75V7.75C9.01868 7.94891 8.93966 8.13968 8.79901 8.28033C8.65835 8.42098 8.46759 8.5 8.26868 8.5C8.06976 8.5 7.879 8.42098 7.73835 8.28033C7.59769 8.13968 7.51868 7.94891 7.51868 7.75V4.75C7.51868 4.55109 7.59769 4.36032 7.73835 4.21967C7.879 4.07902 8.06976 4 8.26868 4ZM8.26868 12C8.53389 12 8.78825 11.8946 8.97578 11.7071C9.16332 11.5196 9.26868 11.2652 9.26868 11C9.26868 10.7348 9.16332 10.4804 8.97578 10.2929C8.78825 10.1054 8.53389 10 8.26868 10C8.00346 10 7.74911 10.1054 7.56157 10.2929C7.37403 10.4804 7.26868 10.7348 7.26868 11C7.26868 11.2652 7.37403 11.5196 7.56157 11.7071C7.74911 11.8946 8.00346 12 8.26868 12Z" fill="currentColor"></path>
</svg>
"""

DEFAULT_SYSTEM = """
You are a chatbot designed specifically for use as a voice assistant. Your responses must be clear, precise, and always shorter than 300 characters. Prioritize natural, conversational language suitable for speech. Avoid filler, long explanations, or complex phrasing.
""".strip()

IN_CHANNELS = 1
IN_RATE = 24000
IN_CHUNK = 1024
IN_SAMPLE_WIDTH = 2
VAD_STRIDE = 0.5

OUT_CHANNELS = 1
OUT_RATE = 24000
OUT_SAMPLE_WIDTH = 2
OUT_CHUNK = 5760

OUT_CHUNK = 20 * 4096
OUT_RATE = 24000
OUT_CHANNELS = 1

API = {
    'LLM': {
        'url': os.environ.get('VLLM_LLM_API', 'http://localhost:8001'),
        'status': False
    },
}
DUMMY_AUDIO = os.environ.get('DUMMY_AUDIO', 'true').lower() == 'true'

def check_ping():
    global API

    for k in API.keys():
        try:
            r = requests.get(API[k]['url'] + '/ping', timeout = 1.0)
            API[k]['status'] = r.status_code == 200
        except:
            pass
        time.sleep(1.0)

def check_health():
    while True:
        check_ping()
        
check_ping()
thread = Thread(target=check_health, daemon=True)
thread.start()

def run_vad(ori_audio, sr):
    _st = time.time()
    try:
        audio = ori_audio
        audio = audio.astype(np.float32) / 32768.0
        sampling_rate = 16000
        if sr != sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)

        vad_parameters = {}
        vad_parameters = VadOptions(**vad_parameters)
        speech_chunks = get_speech_timestamps(audio, vad_parameters)
        audio = collect_chunks(audio, speech_chunks)
        duration_after_vad = audio.shape[0] / sampling_rate

        if sr != sampling_rate:
            # resample to original sampling rate
            vad_audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=sr)
        else:
            vad_audio = audio
        vad_audio = np.round(vad_audio * 32768.0).astype(np.int16)
        vad_audio_bytes = vad_audio.tobytes()

        return duration_after_vad, vad_audio_bytes, round(time.time() - _st, 4)
    except Exception as e:
        msg = f"[asr vad error] audio_len: {len(ori_audio)/(sr*2):.3f} s, trace: {traceback.format_exc()}"
        print(msg)
        return -1, ori_audio, round(time.time() - _st, 4)


def warm_up():
    frames = b"\x00\x00" * 1024 * 2  # 1024 frames of 2 bytes each
    dur, frames, tcost = run_vad(frames, 16000)
    print(f"warm up done, time_cost: {tcost:.3f} s")


warm_up()


@dataclass
class AppState:
    stream: np.ndarray | None = None
    sampling_rate: int = 0
    pause_detected: bool = False
    started_talking: bool =  False
    stopped: bool = False
    conversation: list = field(default_factory=list)


def determine_pause(audio: np.ndarray, sampling_rate: int, state: AppState) -> bool:
    """Take in the stream, determine if a pause happened"""

    temp_audio = audio
    
    dur_vad, _, time_vad = run_vad(temp_audio, sampling_rate)
    duration = len(audio) / sampling_rate

    if dur_vad > 0.5 and not state.started_talking:
        print("started talking")
        state.started_talking = True
        return False

    print(f"duration: {duration:.3f} s, duration_after_vad: {dur_vad:.3f} s, time_vad: {time_vad:.3f} s")

    return (duration - dur_vad) > 0.5

def chat(
    conversation, 
    temperature, 
    top_p, 
    max_output_tokens, 
    system_prompt,
):
    messages = [{'role': 'system', 'content': system_prompt}]
    user = True
    for c in conversation[-10:]:
        print(c)
        if c['role'] == 'assistant' and isinstance(c['content'], dict):
            continue
        if c['role'] == 'user' and user:
            with open(c['content']['path'], 'rb') as fopen:
                audio_base64 = base64.b64encode(fopen.read()).decode('utf-8')
            data = {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_base64,
                            "format": "wav"
                        },
                    },
                ],
            }
            user = False
        else:
            data = {
                'role': 'assistant', 'content':c['content']
            }
            user = True
        messages.append(data)

    data = {
        'model': 'mesolitica/Malaysian-Qwen2.5-7B-Speech-Instruct',
        'max_completion_tokens': max_output_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'messages': messages,
        'stream': True,
    }
    with requests.post(API['LLM']['url'] + '/v1/chat/completions', json=data, stream=True) as response:
        try:
            for line in response.iter_lines():
                if line:
                    if line.startswith(b"data: "):
                        line = line[len(b"data: "):]
                    if line == b"[DONE]":
                        break
                    content = json.loads(line)
                    delta = content['choices'][0]['delta'].get('content')
                    if delta:
                        yield delta
        except Exception as e:
            print(e)
            pass
        
def speaking(text):
    response = elevenlabs.text_to_speech.stream(
        voice_id="NpVSXJvYSdIbjOaMbShj",
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=1.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=False,
            speed=1.0,
        ),
    )
    audio_stream = io.BytesIO()
    for chunk in response:
        if chunk:
            audio_stream.write(chunk)
    audio_stream.seek(0)
    return sf.read(audio_stream)
    

def start_recording(state: AppState):
    if not API['LLM']['status']:
        state.pause_detected = False
        state.started_talking = False
        gr.Warning(f"LLM API is not available, please try again later.", duration=5)
        return gr.Audio(recording=False), state
        
    return None, state

def process_audio(audio: tuple, state: AppState):
    if state.stream is None:
        state.stream = audio[1]
        state.sampling_rate = audio[0]
    else:
        state.stream =  np.concatenate((state.stream, audio[1]))

    pause_detected = determine_pause(state.stream, state.sampling_rate, state)
    state.pause_detected = pause_detected

    if state.pause_detected and state.started_talking:
        print('pause detected')
        state.started_talking = False
        return gr.Audio(recording=False), state
    return None, state


def response(
    state: AppState, 
    temperature, 
    top_p, 
    max_output_tokens, 
    chunk_size, 
    system_prompt,
):
    if not state.pause_detected and not state.started_talking:
        return None, AppState()
    
    audio_buffer = io.BytesIO()

    segment = AudioSegment(
        state.stream.tobytes(),
        frame_rate=state.sampling_rate,
        sample_width=state.stream.dtype.itemsize,
        channels=(1 if len(state.stream.shape) == 1 else state.stream.shape[1]),
    )
    segment.export(audio_buffer, format="wav")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_buffer.getvalue())
    
    state.conversation.append({
        "role": "user",
        "content": {
            "path": f.name,
            "mime_type": "audio/wav"
        }
    })

    audio = []
    first_time = True
    generated_text = ''
    temp = ''
    
    for new_text in chat(
        conversation=state.conversation,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        system_prompt=system_prompt,
    ):
        temp += new_text
        t = temp.strip()
        if len(t.split()) >= chunk_size or (len(t) and t[-1] in ',.?'):
            y, sr = speaking(temp)
            generated_text += temp
            audio.append(y)
            
            yield (sr, y), generated_text, state
            temp = ''
    
    if len(temp):
        y, sr = speaking(temp)
        generated_text += temp
        audio.append(y)

        yield (sr, y), generated_text, state

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        sf.write(f, np.concatenate(audio), 22050)
    
    state.conversation.append({
        "role": "assistant",
        "content": {
            "path": f.name,
            "mime_type": "audio/mp3"
            }
        })
    state.conversation.append({"role": "assistant", "content": generated_text})
    yield None, generated_text, AppState(conversation=state.conversation)


status = ALIVE if API['LLM']['status'] else NOT_ALIVE

with gr.Blocks() as demo:
    gr.HTML(f"""
<div style="display: flex; align-items: center; justify-content: space-between;">
  <div style="display: flex; align-items: center;">
    <img src="https://mesolitica.com/images/mesolitica-transparent.png" style="height: 30px; margin-right: 8px;"/>
    <span>Malaysian Real-time Speech-to-Speech</span>
  </div>
  <div style="display: flex; align-items: center;">
    <span style="padding-right: 5px">Model status:</span>
    {status}
  </div>
</div>
""")
    with gr.Row():
        with gr.Column():
            input_audio = gr.Audio(
                label="Input Audio", sources="microphone", type="numpy"
            )
            with gr.Accordion("Parameters", open=False) as parameter_row:
                temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Temperature",)
                top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.1, interactive=True, label="Top P",)
                max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max Output Tokens",)
                chunk_size = gr.Slider(minimum=0, maximum=30, value=20, step=1, interactive=True, label="Chunk Word Size",)
                system_prompt = gr.Textbox(label="System Prompt", type="text", lines = 10, max_lines = 10, value = DEFAULT_SYSTEM, interactive = True)
        with gr.Column():
            chatbot = gr.Chatbot(label="Conversation", type="messages")
            output_audio = gr.Audio(label="Output Audio", streaming=True, autoplay=True)
            output_text = gr.Textbox(label="Output Text", type="text")
    state = gr.State(value=AppState())

    input_audio.start_recording(
        start_recording, 
        [state],
        [input_audio, state],
    )
    stream = input_audio.stream(
        process_audio,
        [input_audio, state],
        [input_audio, state],
        stream_every=0.25,
        time_limit=30,
    )
    respond = input_audio.stop_recording(
        response,
        [state, temperature, top_p, max_output_tokens, chunk_size, system_prompt],
        [output_audio, output_text, state]
    )
    respond.then(lambda s: s.conversation, [state], [chatbot])

    cancel = gr.Button("Stop Conversation", variant="stop")
    cancel.click(lambda: (AppState(stopped=True), gr.Audio(recording=False)), None,
                [state, input_audio], cancels=[respond])


demo.launch(server_name="0.0.0.0")
