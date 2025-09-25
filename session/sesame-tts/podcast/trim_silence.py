import os
import soundfile as sf
import librosa
import json
import click
import numpy as np
import malaya_speech
from glob import glob
from functools import partial
from multiprocess import Pool
from tqdm import tqdm

def chunks(l, devices):
    chunk_size = len(l) // len(devices)
    remainder = len(l) % len(devices)
    start = 0
    for i in range(len(devices)):
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        yield (l[start:end], devices[i])
        start = end

def new_path(f):
    splitted = f.split('/')
    base_folder = splitted[0] + '_trim'
    splitted = '/'.join([base_folder] + splitted[1:])
    return splitted

def new_path_done(f):
    splitted = f.split('/')
    base_folder = splitted[0] + '_trim_done'
    splitted = '/'.join([base_folder] + splitted[1:])
    return splitted

def loop(indices_device_pair):
    files, device = indices_device_pair
    
    vad = malaya_speech.vad.webrtc(minimum_amplitude = 0)
    min_length = 0.4

    for file in tqdm(files):
        folder = os.path.split(file)[0]
        folder_folder = os.path.split(folder)[1]
        f_new = new_path(file)
        filename_done = new_path_done(file)

        try:
            with open(filename_done) as fopen:
                json.load(fopen)
                continue
        except:
            pass
            
        try:
            vad = malaya_speech.vad.webrtc(minimum_amplitude = 0)
            y, sr = librosa.load(file, sr = None)
            start_silent_trail = int(0.3 * sr)
            middle_silent_trail = int(min_length * sr / 2)
            middle_silent_trail, start_silent_trail
            y_= malaya_speech.resample(y, sr, 16000)
            y_ = malaya_speech.astype.float_to_int(y_)
            frames = malaya_speech.generator.frames(y, 30, sr)
            frames_ = list(malaya_speech.generator.frames(y_, 30, 16000, append_ending_trail = False))
            frames_webrtc = [(frames[no], vad(frame)) for no, frame in enumerate(frames_)]
            grouped_deep = malaya_speech.group.group_frames(frames_webrtc)
            r = []
            for no, g in enumerate(grouped_deep):
                if g[1]:
                    g = g[0].array
                else:
                    if no == 0:
                        g = g[0].array[-start_silent_trail:]
                    elif no == (len(grouped_deep) - 1):
                        g = g[0].array[:start_silent_trail]
                    else:
                        if g[0].duration >= min_length:
                            g = [g[0].array[:middle_silent_trail], g[0].array[-middle_silent_trail:]]
                            g = np.concatenate(g)
                        else:
                            g = g[0].array

                r.append(g)
            y_after = np.concatenate(r)
            
            os.makedirs(os.path.split(f_new)[0], exist_ok = True)
            sf.write(f_new, y_after, sr)
            os.makedirs(os.path.split(filename_done)[0], exist_ok = True)
            with open(filename_done, 'w') as fopen:
                json.dump('done', fopen)
            
        except Exception as e:
            print(e)

@click.command()
@click.option('--file')
@click.option('--replication', default = 1)
def main(
    file, 
    replication,
):
    devices = replication * [0]
    
    with open(file) as fopen:
        files = json.load(fopen)
    filtered = []
    for file in tqdm(files):
        filename_done = new_path_done(file)

        if os.path.exists(filename_done):
            try:
                with open(filename_done) as fopen:
                    json.load(fopen)
                    continue
            except:
                pass
                
        filtered.append(file)
    
    df_split = list(chunks(filtered, devices))

    loop_partial = partial(loop)

    with Pool(len(devices)) as pool:
        pooled = pool.map(loop_partial, df_split)

if __name__ == '__main__':
    main()

    