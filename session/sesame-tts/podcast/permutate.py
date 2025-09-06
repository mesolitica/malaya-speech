import os
import soundfile as sf
import json
import click
import re
import random
import numpy as np
from glob import glob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import partial
from collections import defaultdict
from multiprocess import Pool
from tqdm import tqdm
import time

def chunks(l, devices):
    chunk_size = len(l) // len(devices)
    remainder = len(l) % len(devices)
    start = 0
    for i in range(len(devices)):
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        yield (l[start:end], devices[i])
        start = end

rejected = [
    'terima kasih kerana menonton',
    'terima kasih',
    'thank you for watching',
]

def new_path(f):
    return f.replace('_processed/', '_processed_permutate/').replace('.mp3', '.data')

def loop(indices_device_pair):
    files, device = indices_device_pair
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    
    import torch

    torch.set_grad_enabled(False)

    import malaya_speech

    model = malaya_speech.speaker_vector.nemo('huseinzol05/nemo-titanet_large')
    model = model.eval()
    _ = model.cuda()

    for file in tqdm(files):
        folder = os.path.split(file)[0]
        folder_folder = os.path.split(folder)[1]
        filename_done = new_path(file)

        try:
            with open(filename_done) as fopen:
                json.load(fopen)
                continue
        except:
            pass

        try:
            with open(file) as fopen:
                d = json.load(fopen)
        except:
            continue
            
        speakers = defaultdict(dict)
        
        audio_files = []
        index = 0
        for no, obj in enumerate(d):
            text = obj["text"].strip()
            
            rt_ = re.sub('[^a-z ]+', '', text.lower()).strip()
            if any([s == rt_ for s in rejected]):
                continue
                
            split = text.split()
            ones = [w for w in split if len(w) <= 1]
            if (len(ones) / len(split)) >= 0.5:
                continue
                
            if any([(len(set(w)) / len(w)) < 0.3 for w in split]):
                continue
            
            try:
                dense = CountVectorizer(ngram_range = (3,3)).fit_transform([text]).todense()
                repeat = (dense > 3).sum() >= 1
                if repeat:
                    continue
            except:
                continue
            
            audio_path = os.path.join(folder, f'{folder_folder}_{no}.mp3')
            
            if not os.path.exists(audio_path):
                continue
            
            speakers[obj['speaker']][index] = {
                'audio': audio_path,
                'transcription': text,
            }
            audio_files.append(audio_path)
            index += 1
        
        ys = [malaya_speech.load(f)[0] for f in audio_files]
        if not len(ys):
            continue
            
        vectors = []
        for i in range(0, len(ys), 4):
            vectors_ = model(ys[i: i + 4])
            vectors.append(vectors_)
            
        cosine = cosine_similarity(np.concatenate(vectors))
        data = []
        
        for speaker in speakers.keys():
            data_ = []
            for row in speakers[speaker]:
                for row_ in speakers[speaker]:
                    if row == row_:
                        continue
                    
                    if cosine[row, row_] < 0.8:
                        continue

                    data_.append({
                        'reference_audio': speakers[speaker][row]['audio'],
                        'reference_text': speakers[speaker][row]['transcription'],
                        'target_audio': speakers[speaker][row_]['audio'],
                        'target_text': speakers[speaker][row_]['transcription'],
                    })

            data.extend(random.sample(data_, min(len(data_), 30)))

        os.makedirs(os.path.split(filename_done)[0], exist_ok = True)
        with open(filename_done, 'w') as fopen:
            json.dump(data, fopen)

@click.command()
@click.option('--pattern', default = '*_processed/**/*.json')
@click.option('--replication', default = 1)
def main(
    pattern, 
    replication,
):
    devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if devices is None:
        
        import torch
        devices = list(range(torch.cuda.device_count()))
    else:
        devices = [d.strip() for d in devices.split(',')]

    devices = replication * devices
    print(devices)
    
    filtered = []
    files = glob(pattern, recursive = True)
    for file in tqdm(files):
        filename_done = new_path(file)

        try:
            with open(filename_done) as fopen:
                json.load(fopen)
                continue
        except:
            filtered.append(file)
    
    df_split = list(chunks(filtered, devices))

    loop_partial = partial(loop)

    with Pool(len(devices)) as pool:
        pooled = pool.map(loop_partial, df_split)

if __name__ == '__main__':
    main()

    