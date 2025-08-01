from dia.model import Dia
from transformers import pipeline
from torchaudio.functional import resample
from glob import glob
import torch.nn.functional as F
import tempfile
import numpy as np
import time
import gradio as gr
import pathlib
import torch
import torchaudio
import os
import re
import logging

theme = gr.themes.Base()

model = None
asr_pipe = None

MODEL_NAME = os.environ.get('MODEL_NAME', 'mesolitica/Malaysian-Podcast-Dia-1.6B')
maxlen_text = int(os.environ.get('MAXLEN_TEXT', '1000'))
maxlen = 20000
maxlen_str = f'{maxlen // 1000} seconds'
warmup = {}

examples = []
for f in glob('*.mp3'):
    examples.append([f, '', 'Model Text to Speech TTS ini dibangunkan seratus peratus oleh Mesolitica, syarikat pemula di Malaysia yang membangunkan juga Malaysia Large Language Model mallam.', False, 0.15, 1.0])
    
def apply_fade(audio, fade_len=200):
    if len(audio) < fade_len * 2:
        return audio
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    audio[:fade_len] *= fade_in
    audio[-fade_len:] *= fade_out
    return audio

def add_silence_between_chunks(chunks, silence_duration=0.05, sample_rate=16000):
    """
    Adds silence between audio chunks except before the first one.

    Args:
        chunks (List[np.ndarray]): List of 1D audio arrays.
        silence_duration (float): Duration of silence in seconds.
        sample_rate (int): Sample rate of the audio.

    Returns:
        np.ndarray: Concatenated audio with silences in between.
    """
    silence = np.zeros(int(silence_duration * sample_rate), dtype=np.float32)
    output = []

    for i, chunk in enumerate(chunks):
        if i > 0:
            output.append(silence)
        output.append(chunk)

    return np.concatenate(output)

def chunk_text(text, max_chars=135, min_chars=40):
    """
    Splits the input text into chunks, each with a maximum number of characters.

    Args:
        text (str): The text to be split.
        max_chars (int): The maximum number of characters per chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    current_chunk = ""
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())
        
    merged, temp = [], []
    l = 0
    
    for c in chunks:
        temp.append(c)
        l += len(c)
        if l >= min_chars:
            merged.append(' '.join(temp))
            l = 0
            temp = []
    
    if len(temp):
        merged.append(' '.join(temp))
    
    return merged
    
def load_asr_pipe():
    global asr_pipe
    gr.Info('Loading Whisper ASR model.')
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=torch.bfloat16,
        device='cuda',
    )

def load_tts():
    global model
    gr.Info('Loading TTS model.')
    model = Dia.from_pretrained(MODEL_NAME, compute_dtype="float16")

def hotload():
    if model is None:
        load_tts()

    if asr_pipe is None:
        load_asr_pipe()

def basic_tts(
    ref_audio_input,
    ref_text_input,
    gen_text_input,
):
    global warmup

    """
    ref_audio_input: /tmp/gradio/ffdca1484a4fd01840cb3ed4aaa7f2da0433f188170e0215385370af4a162d14/audio.wav
    """
    if ref_audio_input is None:
        raise gr.Error(f"Reference audio cannot be empty.")
    
    if ref_text_input is not None and isinstance(ref_text_input, str) and 1 < len(ref_text_input) < 3:
        raise gr.Error(f"Reference text is too short.")

    if len(gen_text_input) < 2:
        raise gr.Error(f"Text to generate is too short.")

    if len(gen_text_input) >= maxlen_text:
        raise gr.Error(f"Text to generate is more than {maxlen_text} characters, for fair public use, we limit it, feel free to self host.")

    hotload()

    torch.cuda.empty_cache()
    
    y, sr = torchaudio.load(ref_audio_input)
    y = y.mean(dim=0)
    
    if len(ref_text_input) < 2:
        gr.Info('Transcribing reference audio.')
        r_asr = asr_pipe(
            [resample(y, orig_freq=sr, new_freq=16000).numpy()],
            chunk_length_s=30,
            batch_size=8,
            generate_kwargs={"task": "transcribe"},
            return_timestamps=False,
        )
        ref_text_input = r_asr[0]['text'].strip()

    gr.Info('Generating synthetic audio.')
    
    sr = 44100
    ref_text = ref_text_input
    text = gen_text_input
    clone_from_text = f"[S1] {ref_text}"
    clone_from_audio = ref_audio_input

    text = [t.strip() for t in text.split('\n') if len(t.strip())]
    gen_text_batches = []
    for t in text:
        gen_text_batches.extend(chunk_text(t, max_chars=200))
    print(gen_text_batches)
    
    y = []
    for t in gen_text_batches:
        texts = [clone_from_text + '[S1] ' + t.strip()]
        clone_from_audios = [clone_from_audio]

        if not warmup:
            gr.Info('First time generation a bit slow, after that it will be fast.')

        output = model.generate(
            texts, 
            audio_prompt=clone_from_audios, 
            use_torch_compile=True, 
            verbose=True, 
            max_tokens=2500, 
            temperature=1.0, 
            cfg_scale=1.0,
        )
        y.append(output)
        warmup = True

    y = add_silence_between_chunks(y, silence_duration=0.05, sample_rate=sr)
    
    return [(sr, y), ref_text_input]

with gr.Blocks(theme=theme) as demo:
    gr.Markdown(
f"""
# 🇲🇾 Malaysian Voice Cloning

This is a local web gradio UI for Malaysian Voice Cloning.

The model should able to zero-shot any Malaysian and Singaporean speakers.

If you're having issues, try converting your reference audio to WAV or MP3, clipping it to {maxlen_str} with ✂ in the bottom right corner (otherwise might have non-optimal auto-trimmed result).

**Reference text will be automatically transcribed with Whisper if not provided. For best results, keep your reference clips shorter than {maxlen_str}. Ensure the audio is fully uploaded before generating.**

**For fair public use, we limit the maximum to {maxlen_text} characters only.**

**For better output, please normalize numbering, for an example `1000` to `one thousand` or `satu ribu`.**
"""
    )
    ref_audio_input = gr.Audio(label="Reference Audio", type="filepath")
    ref_text_input = gr.Textbox(
        label="Reference Text",
        info="Leave blank to automatically transcribe the reference audio. If you enter text it will override automatic transcription.",
        lines=3,
    )
    gen_text_input = gr.Textbox(
        label="Text to Generate", 
        info="the model is more accurate on longer text.",
        lines=3,
    )
    audio_output = gr.Audio(label="Synthesized Audio", show_download_button = True)
    generate_btn = gr.Button("Synthesize", variant="primary")
    
    generate_btn.click(
        basic_tts,
        inputs=[
            ref_audio_input,
            ref_text_input,
            gen_text_input,
        ],
        outputs=[audio_output, ref_text_input],
    )
    examples = gr.Examples(
        examples=examples,
        inputs=[
            ref_audio_input,
            ref_text_input,
            gen_text_input,
        ],
    )

if __name__ == "__main__":
    if os.environ.get('HOTLOAD', 'false').lower() == 'true':
        print('hotloading the models')
        hotload()

    demo.launch(server_name="0.0.0.0")