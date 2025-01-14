import malaya_speech
import numpy as np
import gradio as gr
import modelscope_studio as mgr
import os
import torch
import time

tts = None
def load_tts():
    global tts
    tts = malaya_speech.tts.vits(model = 'mesolitica/VITS-osman')
    _ = tts.cuda()
    print('done load TTS')

def hotload():
    load_tts()
    pass

def add_speech(speech, text, temperature, top_p, max_output_tokens, chunk_size):
    text = text.strip().split()
    text_list = []
    with torch.no_grad():
        t = []
        for i in range(len(text)):
            t.append(text[i])
            if len(t) >= chunk_size or t[-1][-1] in ',.?':
                t = ' '.join(t)
                r_osman = tts.predict(t)
                text_list.append(t)
            
                yield ' '.join(text_list), (22050, r_osman['y'])
                time.sleep(0.5)
                t = []

        if len(t):
            t = ' '.join(t)
            r_osman = tts.predict(t)
            text_list.append(t)
        
            yield ' '.join(text_list), (22050, r_osman['y'])
            time.sleep(0.0)



def clear_history():
    return None, '', None

with gr.Blocks() as demo:
    gr.Markdown("""<p align="center"><img src="https://mesolitica.com/images/mesolitica-transparent.png" style="height: 60px"/><p>""")
    gr.Markdown("""<center><font size=4>🇲🇾 Malaysian Speech-to-Speech</center>""")
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

    submit_btn.click(
        add_speech,
        [audio_input_box, text_input_box, temperature, top_p, max_output_tokens, chunk_size],
        [text_output_box, audio_output_box]
    )

    clear_btn.click(
        clear_history,
        None,
        [audio_input_box, text_output_box, audio_output_box],
        queue=False
    )

if __name__ == "__main__":
    if os.environ.get('HOTLOAD', 'false').lower() == 'true':
        print('hotloading the models')
        hotload()

    demo.queue().launch(server_name="0.0.0.0")
