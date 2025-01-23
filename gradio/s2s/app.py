import malaya_speech
import numpy as np
import gradio as gr
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, TextIteratorStreamer
import os
import torch
import time
import librosa
import soundfile as sf
from threading import Thread
from datasets import Audio

tts = None
processor = None
model = None
streamer = None

audio_token = "<|AUDIO|>"
audio_bos_token = "<|audio_bos|>"
audio_eos_token = "<|audio_eos|>"
audio_token_id = None
pad_token_id = None

audio = Audio(sampling_rate=16000)

def load_tts():
    global tts

    tts = malaya_speech.tts.vits(model = 'mesolitica/VITS-husein')
    _ = tts.cuda()
    print('done load TTS')

def load_model():
    global processor, model, audio_token_id, pad_token_id, streamer

    processor = AutoProcessor.from_pretrained('Qwen/Qwen2-Audio-7B-Instruct')
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
    'mesolitica/Malaysian-Qwen2-Audio-7B-Instruct', torch_dtype = torch.bfloat16).cuda()

    with torch.no_grad():
        model(input_ids = torch.arange(1024)[None].cuda())

    audio_token_id = processor.tokenizer._convert_token_to_id_with_added_voc('<|AUDIO|>')
    pad_token_id = processor.tokenizer.pad_token_id

    print('done load Model')

def hotload():
    load_tts()
    load_model()

def add_speech(speech, text, temperature, top_p, max_output_tokens, chunk_size):
    print(speech, text, temperature, top_p, max_output_tokens, chunk_size)
    if speech is None and len(text) < 2:
        raise gr.Error(f"Cannot both audio and text empty.")
    
    inputs = []
    if speech is not None:
        inputs.append({"type": "audio", "audio_url": "url"})
    if len(text):
        inputs.append({"type": "text", "text": text})
    
    d = [{"role": "user", "content": inputs}]
    text = processor.apply_chat_template(d, return_tensors = 'pt', add_generation_prompt=True)

    sample = text
    if speech is not None:
        y = speech[1].astype(np.float32) / np.iinfo(np.int16).max
        y = librosa.resample(y, orig_sr = speech[0], target_sr = 16000)

        inputs_audio = processor.feature_extractor([y], return_attention_mask=True, padding="max_length", return_tensors = 'pt')
        audio_lengths = inputs_audio["attention_mask"].sum(-1).tolist()
        input_features = inputs_audio['input_features']
        feature_attention_mask = inputs_audio['attention_mask']
        num_audio_tokens = sample.count(audio_token)
        replace_str = []
        while audio_token in sample:
            audio_length = audio_lengths.pop(0)
            input_length = (audio_length - 1) // 2 + 1
            num_audio_tokens = (input_length - 2) // 2 + 1

            expanded_audio_token = audio_token * num_audio_tokens

            audio_token_start_idx = sample.find(audio_token)
            audio_token_end_idx = audio_token_start_idx + len(audio_token)

            has_bos = (
                sample[audio_token_start_idx - len(audio_bos_token) : audio_token_start_idx]
                == audio_bos_token
            )
            has_eos = (
                sample[audio_token_end_idx : audio_token_end_idx + len(audio_eos_token)]
                == audio_eos_token
            )

            if not has_bos and not has_eos:
                expanded_audio_token = audio_bos_token + expanded_audio_token + audio_eos_token

            replace_str.append(expanded_audio_token)
            sample = sample.replace(audio_token, "<placeholder>", 1)

        while "<placeholder>" in sample:
            sample = sample.replace("<placeholder>", replace_str.pop(0), 1)

    else:
        input_features = None
        feature_attention_mask = None
    
    inputs = processor.tokenizer(sample, return_tensors = 'pt').to('cuda')
    inputs['input_features'] = input_features
    inputs['feature_attention_mask'] = feature_attention_mask

    streamer = TextIteratorStreamer(processor.tokenizer)
    generation_kwargs = dict(
        max_new_tokens=max_output_tokens,
        top_p=top_p,
        top_k=20,
        temperature=temperature,
        do_sample=True,
        repetition_penalty=1.1,
        streamer=streamer,
        **inputs
    )
    with torch.no_grad():
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        generated_text = ''
        temp = ''
        for new_text in streamer:
            if new_text == sample:
                continue

            new_text = new_text.replace('<|im_end|>', '')

            temp += new_text
            t = temp.strip()

            if len(t.split()) >= chunk_size or (len(t) and t[-1] in ',.?'):
                r_osman = tts.predict(temp)
                generated_text += temp
            
                yield generated_text, (22050, r_osman['y'])
                temp = ''

        if len(temp):
            r_osman = tts.predict(temp)
            generated_text += temp
        
            yield generated_text, (22050, r_osman['y'])
            time.sleep(0.0)

def clear_history():
    return None, '', None

with gr.Blocks() as demo:
    gr.Markdown("""<p align="center"><img src="https://mesolitica.com/images/mesolitica-transparent.png" style="height: 60px"/><p>""")
    gr.Markdown("""<center><font size=5>🇲🇾 Malaysian Speech-to-Speech</center>""")
    gr.Markdown(
        """\
<center><font size=3>This model is still in early stage development, the respond might be not perfect, but we can make it better!</center>""")
    with gr.Row():
        with gr.Column():
            audio_input_box = gr.Audio(sources=["upload", "microphone"], label="Speech Input")
            text_input_box = gr.Textbox(label="Text Input")
        with gr.Accordion("Parameters", open=True) as parameter_row:
            temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.1, interactive=True, label="Temperature",)
            top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P",)
            max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max Output Tokens",)
            chunk_size = gr.Slider(minimum=0, maximum=30, value=20, step=1, interactive=True, label="Chunk Word Size",)
    
    with gr.Row():
        submit_btn = gr.Button(value="Send", variant="primary")
        clear_btn = gr.Button(value="Clear")

    text_output_box = gr.Textbox(label="Text Output", type="text")
    audio_output_box = gr.Audio(
        label="Speech Output", show_download_button = True, editable = False,
        streaming=True, autoplay=True,
    )

    click_event = submit_btn.click(
        add_speech,
        [audio_input_box, text_input_box, temperature, top_p, max_output_tokens, chunk_size],
        [text_output_box, audio_output_box]
    )

    clear_btn.click(
        clear_history,
        None,
        [audio_input_box, text_output_box, audio_output_box],
        queue=False,
        cancels=[click_event]
    )

if __name__ == "__main__":
    if os.environ.get('HOTLOAD', 'false').lower() == 'true':
        print('hotloading the models')
        hotload()

    demo.queue().launch(server_name="0.0.0.0")
