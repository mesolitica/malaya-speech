from dynamicbatch_ttspipeline.f5_tts.load import (
    load_f5_tts,
    load_vocoder,
    target_sample_rate,
    hop_length,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
)
from dynamicbatch_ttspipeline.f5_tts.utils import (
    chunk_text,
    convert_char_to_pinyin,
)
from dynamicbatch_ttspipeline.resemble_enhance.enhancer.inference import load_enhancer
from dynamicbatch_ttspipeline.resemble_enhance.inference import (
    remove_weight_norm_recursively,
    merge_chunks,
)
from torch.nn.utils.rnn import pad_sequence
from torchaudio.functional import resample
from pydub import AudioSegment, silence
from transformers import pipeline
import torch.nn.functional as F
from glob import glob
import tempfile
import numpy as np
import time
import gradio as gr
import pathlib
import torch
import torchaudio
import os

theme = gr.themes.Base()

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
torch_dtype = torch.bfloat16

vocoder = None
model = None
asr_pipe = None
speech_enhancement = None
npad = 441
hp = None
speech_enhancement_sr = None
chunk_length = None
overlap_length = None
speech_enhancement_hop_length = None

maxlen = 20000
maxlen_str = f'{maxlen // 1000} seconds'

examples = []
for f in glob('*.mp3'):
    examples.append([f, '', 'Model Text to Speech TTS ini dibangunkan seratus peratus oleh Mesolitica, syarikat pemula di Malaysia yang membangunkan juga Malaysia Large Language Model mallam.', False, True, 0.15, 1.0])

def load_speech_enhancement():
    global speech_enhancement, hp, speech_enhancement_sr, chunk_length, overlap_length, speech_enhancement_hop_length
    gr.Info('Loading Speech Enhancement model.')
    speech_enhancement = load_enhancer(run_dir = None, device = device, dtype = torch.float32)
    speech_enhancement.configurate_(nfe=64, solver='midpoint', lambd=0.9, tau=0.5)
    remove_weight_norm_recursively(speech_enhancement)
    speech_enhancement.normalizer.eval()
    hp = speech_enhancement.hp
    speech_enhancement_sr = hp.wav_rate
    chunk_seconds = 10.0
    overlap_seconds = 1.0
    chunk_length = int(speech_enhancement_sr * chunk_seconds)
    overlap_length = int(speech_enhancement_sr * overlap_seconds)
    speech_enhancement_hop_length = chunk_length - overlap_length

def load_asr_pipe():
    global asr_pipe
    gr.Info('Loading Whisper ASR model.')
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=torch_dtype,
        device=device,
    )

def load_tts():
    global model, vocoder
    gr.Info('Loading TTS model.')
    model = load_f5_tts('mesolitica/Malaysian-F5-TTS', device = device, dtype = torch.float16)
    vocoder = load_vocoder('mesolitica/malaysian-vocos-mel-24khz', device = device)
    convert_char_to_pinyin(['helo'])


def speech_enhancement_func(y, sr, resample_back = True):
    y = resample(
        y,
        orig_freq=sr,
        new_freq=speech_enhancement_sr,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )
    
    with torch.no_grad():
        audios, lengths, abs_maxes = [], [], []
        for start in range(0, y.shape[-1], speech_enhancement_hop_length):
            chunk = y[start : start + chunk_length]
            lengths.append(chunk.shape[-1])
            abs_max = chunk.abs().max().clamp(min=1e-7)
            abs_maxes.append(abs_max)
            chunk = chunk.type(torch.float32)
            chunk = chunk / abs_max
            chunk = F.pad(chunk, (0, npad))
            audios.append(chunk)

        results = []
        batch_size = 3
        for i in range(0, len(audios), batch_size):
            b = audios[i: i + batch_size]
            l = lengths[i: i + batch_size]
            a = abs_maxes[i: i + batch_size]
            padded = pad_sequence(b, batch_first=True).to(device)
            hwav = speech_enhancement(padded).cpu()
            results.extend([hwav[k][:l[k]] * a[k] for k in range(len(hwav))])

        hwav = merge_chunks(
            results, 
            chunk_length, 
            speech_enhancement_hop_length, 
            sr=speech_enhancement_sr, 
            length=y.shape[-1]
        )
    if resample_back:
        return resample(hwav, orig_freq=speech_enhancement_sr, new_freq=sr)
    else:
        return hwav

def remove_silence_edges(audio, silence_threshold=-42):
    non_silent_start_idx = silence.detect_leading_silence(audio, silence_threshold=silence_threshold)
    audio = audio[non_silent_start_idx:]

    non_silent_end_duration = audio.duration_seconds
    for ms in reversed(audio):
        if ms.dBFS > silence_threshold:
            break
        non_silent_end_duration -= 0.001
    trimmed_audio = audio[: int(non_silent_end_duration * 1000)]

    return trimmed_audio

def hotload():
    if model is None:
        load_tts()

    if asr_pipe is None:
        load_asr_pipe()

    if speech_enhancement is None:
        load_speech_enhancement()

def basic_tts(
    ref_audio_input,
    ref_text_input,
    gen_text_input,
    reference_enhancement,
    output_enhancement,
    cross_fade_duration_slider,
    speed_slider,
):
    """
    ref_audio_input: /tmp/gradio/ffdca1484a4fd01840cb3ed4aaa7f2da0433f188170e0215385370af4a162d14/audio.wav
    """
    if ref_audio_input is None:
        raise gr.Error(f"Reference audio cannot be empty.")
    
    if ref_text_input is not None and isinstance(ref_text_input, str) and 1 < len(ref_text_input) < 3:
        raise gr.Error(f"Reference text is too short.")

    if len(gen_text_input) < 2:
        raise gr.Error(f"Text to generate is too short.")

    torch.cuda.empty_cache()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        aseg = AudioSegment.from_file(ref_audio_input)
        non_silent_segs = silence.split_on_silence(
            aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10
        )
        non_silent_wave = AudioSegment.silent(duration=0)
        for non_silent_seg in non_silent_segs:
            if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > maxlen:
                gr.Info(f"Audio is over {maxlen_str}, clipping short.")
                break
            non_silent_wave += non_silent_seg
        if len(non_silent_wave) > maxlen:
            non_silent_segs = silence.split_on_silence(
                aseg, min_silence_len=100, silence_thresh=-40, keep_silence=1000, seek_step=10
            )
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > maxlen:
                    gr.Info(f"Audio is over {maxlen_str}, clipping short.")
                    break
                non_silent_wave += non_silent_seg
        aseg = non_silent_wave
        if len(aseg) > maxlen:
            aseg = aseg[:maxlen]
            gr.Info(f"Audio is over {maxlen_str}, clipping short.")
        
        aseg = remove_silence_edges(aseg) + AudioSegment.silent(duration=100)
        aseg.export(f.name, format="wav")
        print(ref_audio_input, f.name)
        ref_audio_input = f.name
    
    y, sr = torchaudio.load(ref_audio_input)
    y = y.mean(dim=0)

    hotload()

    if reference_enhancement:
        y = speech_enhancement_func(y, sr)
    
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
    
    audio = resample(y, orig_freq=sr, new_freq=target_sample_rate)[None].to(device)
    sr = target_sample_rate
    ref_text = ref_text_input
    text = gen_text_input
    speed = speed_slider
    cross_fade_duration = cross_fade_duration_slider
    
    max_chars = int(len(ref_text.encode("utf-8")) / (audio.shape[-1] / sr) * (25 - audio.shape[-1] / sr))
    text = [t.strip() for t in text.split('\n') if len(t.strip())]
    gen_text_batches = []
    for t in text:
        gen_text_batches.extend(chunk_text(t, max_chars=max_chars))
    print(gen_text_batches)

    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "
    
    ref_audio_len = audio.shape[-1] // hop_length
    final_text_lists, durations, after_durations = [], [], []
    results = []
    for gen_text in gen_text_batches:
        text_list = [ref_text + gen_text]
        final_text_list = convert_char_to_pinyin(text_list)
        ref_text_len = len(ref_text.encode("utf-8"))
        gen_text_len = len(gen_text.encode("utf-8"))
        after_duration = int(ref_audio_len / ref_text_len * gen_text_len / speed)
        final_text_lists.append(final_text_list[0])
        dur = ref_audio_len + after_duration
    
        with torch.no_grad():
            generated, _ = model.sample(
                cond=audio,
                text=final_text_list,
                duration=torch.Tensor([dur]).to(device).type(torch.long),
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )
            generated = generated.to(torch.float32)
            generated = generated[:, ref_audio_len:, :]
            generated_mel_spec = generated.permute(0, 2, 1)
            generated_wave = vocoder.decode(generated_mel_spec)
            generated_wave = generated_wave.cpu().numpy()
            results.append(generated_wave[0])

    if cross_fade_duration <= 0:
        y = np.concatenate(results)
    else:
        final_wave = results[0]
        for i in range(1, len(results)):
            prev_wave = final_wave
            next_wave = results[i]
            cross_fade_samples = int(cross_fade_duration * target_sample_rate)
            cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))
            if cross_fade_samples <= 0:
                final_wave = np.concatenate([prev_wave, next_wave])
                continue
            
            prev_overlap = prev_wave[-cross_fade_samples:]
            next_overlap = next_wave[:cross_fade_samples]
            fade_out = np.linspace(1, 0, cross_fade_samples)
            fade_in = np.linspace(0, 1, cross_fade_samples)
            cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in
            new_wave = np.concatenate(
                [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
            )
            final_wave = new_wave
        y = final_wave

    if output_enhancement:
        y = speech_enhancement_func(torch.tensor(y), sr, resample_back = False)
        sr = speech_enhancement_sr
        y = y.numpy()

    audio = (sr, y)
    return [audio, ref_text_input]

with gr.Blocks(theme=theme) as demo:
    gr.Markdown(
f"""
# 🇲🇾 Malaysian Voice Cloning
This is a local web gradio UI for Malaysian Voice Cloning with advanced batch processing support. This app supports the following TTS model:
* [F5-TTS](https://arxiv.org/abs/2410.06885) (A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching)

The model should able to zero-shot any Malaysian and Singaporean speakers.

If you're having issues, try converting your reference audio to WAV or MP3, clipping it to {maxlen_str} with  ✂  in the bottom right corner (otherwise might have non-optimal auto-trimmed result).

**Reference text will be automatically transcribed with Whisper if not provided. For best results, keep your reference clips shorter than {maxlen_str}. Ensure the audio is fully uploaded before generating.**
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
    reference_enhancement = gr.Checkbox(
        label="Reference Enhancement",
        info="Apply Speech Enhancement to reduce noise for reference audio, this will also increase generation time.",
        value=False,
    )
    output_enhancement = gr.Checkbox(
        label="Output Enhancement",
        info="Apply Speech Enhancement to reduce noise for generated audio, this will also increase generation time.",
        value=True,
    )
    speed_slider = gr.Slider(
        label="Speed",
        minimum=0.3,
        maximum=2.0,
        value=1.0,
        step=0.1,
        info="Adjust the speed of the audio, lower will generate slower audio.",
    )
    cross_fade_duration_slider = gr.Slider(
        label="Cross-Fade Duration (s)",
        minimum=0.0,
        maximum=1.0,
        value=0.15,
        step=0.01,
        info="Set the duration of the cross-fade between audio clips.",
    )
    audio_output = gr.Audio(label="Synthesized Audio", show_download_button = True)
    generate_btn = gr.Button("Synthesize", variant="primary")
    
    generate_btn.click(
        basic_tts,
        inputs=[
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            reference_enhancement,
            output_enhancement,
            cross_fade_duration_slider,
            speed_slider,
        ],
        outputs=[audio_output, ref_text_input],
    )
    examples = gr.Examples(
        examples=examples,
        inputs=[
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            reference_enhancement,
            output_enhancement,
            cross_fade_duration_slider,
            speed_slider,
        ],
    )
    gr.Markdown("""
## Remarks

If the audio generated sounds like little bit underwater, that is probably Vocoder, we are building better Vocoder!

## Source code

Source code of the Gradio UI at [malaya-speech/gradio/f5-tts](https://github.com/mesolitica/malaya-speech/tree/master/gradio/f5-tts).

## Checkpoints

All checkpoints including optimizer states at [mesolitica/Malaysian-F5-TTS](https://huggingface.co/mesolitica/Malaysian-F5-TTS).

## Dataset

We trained on **Malaysian Emilia: An Extensive, Multilingual, and Diverse Speech Dataset for Large-Scale Malaysian Speech Generation**, we open source it at [mesolitica/Malaysian-Emilia](https://huggingface.co/datasets/mesolitica/Malaysian-Emilia).

## Dynamic batching

We created a library to do dynamic batching with Torch compile to serve better concurrency with speed, check it out at [mesolitica/dynamic-batch-TTS-pipeline](https://github.com/mesolitica/dynamic-batch-TTS-pipeline).
""")


if __name__ == "__main__":
    if os.environ.get('HOTLOAD', 'false').lower() == 'true':
        print('hotloading the models')
        hotload()

    demo.queue().launch(server_name="0.0.0.0")