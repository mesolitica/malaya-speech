import librosa
import numpy as np
import os
import malaya_speech
import soundfile as sf
from malaya_speech import Pipeline
from datasets import Audio
from tqdm import tqdm
from malaya_speech.utils.text import TextIDS
from glob import glob
from multiprocess import Pool
import json
import random
import json
import click
import pandas as pd
import itertools
import re

_pad = 'pad'
_start = 'start'
_eos = 'eos'
_punctuation = "!'(),.:;? "
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

MALAYA_SPEECH_SYMBOLS = (
    [_pad, _start, _eos] + list(_special) + list(_punctuation) + list(_letters)
)

config = {
    'sampling_rate': 22050,
    'fft_size': 1024,
    'hop_size': 256,
    'win_length': None,
    'window': 'hann',
    'num_mels': 80,
    'fmin': 80,
    'fmax': 7600,
    'global_gain_scale': 1.0,
    'trim_silence': True,
    'trim_threshold_in_db': 60,
    'trim_frame_size': 2048,
    'trim_hop_size': 512,
}


def chunks(l, n):
    for i in range(0, len(l), n):
        yield (l[i: i + n], i // n)


def multiprocessing(strings, function, cores=6, returned=True):
    df_split = chunks(strings, len(strings) // cores)
    pool = Pool(cores)
    pooled = pool.map(function, df_split)
    pool.close()
    pool.join()

    if returned:
        return list(itertools.chain(*pooled))

def is_outlier(x, p25, p75):
    """Check if value is an outlier."""
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)
    return x <= lower or x >= upper


def remove_outlier(x, p_bottom: int = 25, p_top: int = 75):
    """Remove outlier from x."""
    p_bottom = np.percentile(x, p_bottom)
    p_top = np.percentile(x, p_top)

    indices_of_outliers = []
    for ind, value in enumerate(x):
        if is_outlier(value, p_bottom, p_top):
            indices_of_outliers.append(ind)

    x[indices_of_outliers] = 0.0
    x[indices_of_outliers] = np.max(x)
    return x

def tts_encode(string: str, add_eos: bool = False):
    r = [MALAYA_SPEECH_SYMBOLS.index(c) for c in string if c in MALAYA_SPEECH_SYMBOLS]
    if add_eos:
        r = r + [MALAYA_SPEECH_SYMBOLS.index('eos')]
    return r

def process(
    txts, 
    start_silent_trail = int(0.15 * config['sampling_rate']),
    middle_silent_trail = int(0.2 * config['sampling_rate']),
    end_silent_trail = int(0.2 * config['sampling_rate']),
    process_middle_silent = True,
    maxlen = 25,
):
    
    reader = Audio(sampling_rate = 22050)
    vad = malaya_speech.vad.webrtc()
    txts = txts[0]
    audios, text_ids = [], []

    for f in tqdm(txts):
        
        directory = f[2]
        text = f[1]
        f = f[0]
        f = f.replace('normalized/', 'normalized-enhanced/')
        
        if not os.path.exists(f):
            continue
        
        os.makedirs(directory, exist_ok = True)
        
        audio = reader.decode_example(reader.encode_example(f))['array']

        if config['trim_silence']:
            y_= malaya_speech.resample(audio, config['sampling_rate'], 16000)
            y_ = malaya_speech.astype.float_to_int(y_)
            frames = list(malaya_speech.generator.frames(audio, 30, config['sampling_rate']))
            frames_ = list(malaya_speech.generator.frames(y_, 30, 16000, append_ending_trail = False))
            frames_webrtc = [(frames[no], vad(frame)) for no, frame in enumerate(frames_)]
            grouped_deep = malaya_speech.group.group_frames(frames_webrtc)
            grouped_deep = malaya_speech.group.group_frames_threshold(grouped_deep, 0.15)
            r = []
            for no, g in enumerate(grouped_deep):
                if g[1]:
                    g = g[0].array
                else:
                    if no == 0:
                        g = g[0].array[-start_silent_trail:]
                    elif no == (len(grouped_deep) - 1):
                        g = g[0].array[:end_silent_trail]
                    else:
                        if process_middle_silent:
                            g = np.concatenate([g[0].array[:middle_silent_trail], g[0].array[-middle_silent_trail:]])
                        else:
                            g = g[0].array
                        
                r.append(g)
            audio = np.concatenate(r)
        
        if (len(audio) / config['sampling_rate']) > maxlen:
            continue
        
        if (len(audio) / config['sampling_rate']) < 0.5:
            continue
            
        audio = np.pad(audio, (0, config["fft_size"]), mode="edge")
        f = f.replace('/', '-')
        new_f = f'{directory}/{f}'.replace('.mp3', '.wav').replace('.flac', '.wav')
        sf.write(new_f, audio, 22050)
        audios.append(new_f)
        text_ids.append(text)
    
    return [[audios, text_ids]]

@click.command()
@click.option('--file', help='parquet file')
@click.option('--speaker_name', default = 'speaker', help='speaker name')
@click.option('--speaker_id', default = 0, help='speaker id')
@click.option('--batch_size', default = 20000, help='batch size')
@click.option('--cores', default = 20, help='cores')
def main(file, speaker_name, speaker_id, batch_size, cores):
    df = pd.read_parquet(file).to_dict(orient = 'records')
    dataset = []
    for row in tqdm(df):
        dataset.append([row['audio_filename'], tts_encode(row['normalized']), speaker_name])
    
    audios, text_ids, speakers_id = [], [], []
    for i in range(0, len(dataset), batch_size):
        b = dataset[i: i + batch_size]
        results = multiprocessing(b, process, cores = cores, returned = True)
        
        for result in results:
            audios.extend(result[0])
            text_ids.extend(result[1])
            speakers = [os.path.split(r)[0] for r in result[0]]
            speakers_id.extend([speaker_id for s in speakers])
    
    data = []
    for i in tqdm(range(len(audios))):
        data.append((os.path.join(os.getcwd(), audios[i]), speakers_id[i], text_ids[i]))
        
    random.shuffle(data)

    with open(f'multispeaker-clean-vits-{speaker_name}.json', 'w') as fopen:
        json.dump(data, fopen)
    
if __name__ == '__main__':
    main()