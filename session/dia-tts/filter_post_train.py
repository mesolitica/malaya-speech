import os
import json
import click
import torch
import librosa
import numpy as np
import re
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
from glob import glob
from multiprocess import Pool
from tqdm import tqdm

rejected = [
    'terima kasih kerana menonton',
    'terima kasih',
    'thank you for watching',
]

def sequence_1d_np(seq, maxlen=None, padding='post', pad_int=0, return_len=False):
    if padding not in ['post', 'pre']:
        raise ValueError("padding only supported [`post`, `pre`]")

    lengths = np.array([len(s) for s in seq])
    if maxlen is None:
        maxlen = lengths.max()
        
    result = np.full((len(seq), maxlen), pad_int, dtype=seq[0].dtype)

    for i, s in enumerate(seq):
        n = min(len(s), maxlen)
        if padding == 'post':
            result[i, :n] = s[:n]
        else:
            result[i, -n:] = s[:n]

    if return_len:
        return result, lengths.tolist()
    return result

def chunks(l, devices):
    chunk_size = len(l) // len(devices)
    remainder = len(l) % len(devices)
    start = 0
    for i in range(len(devices)):
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        yield (l[start:end], devices[i], i)
        start = end

def loop(files):
    files, device, actual_index = files
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    
    import torch
    import malaya_speech
    torch.set_grad_enabled(False)

    model = malaya_speech.speaker_vector.nemo('huseinzol05/nemo-titanet_large')
    model = model.eval()
    _ = model.cuda()

    data = []
    for file in tqdm(files):
        folder = os.path.split(file)[0]
        folder_folder = os.path.split(folder)[1]

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
            
        ys = [librosa.load(f, sr = 16000)[0] for f in audio_files]
        if not len(ys):
            continue
            
        vectors = []
        for i in range(0, len(ys), 32):
            inputs, lengths = sequence_1d_np(ys[i: i + 32], return_len = True)
            inputs = torch.from_numpy(inputs).cuda()
            lengths = torch.tensor(lengths).cuda()
            o_processor = model.preprocessor(inputs, lengths)
            o_encoder = model.encoder(*o_processor)
            r = model.decoder(*o_encoder)
            vectors.append(r[1].cpu().numpy())

        vectors = np.concatenate(vectors)        
        cosine = cosine_similarity(vectors)
        
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

    with open(f'filter_post2-{actual_index}.json', 'w') as fopen:
        json.dump(data, fopen)
    
@click.command()
@click.option('--folders', default = 'sg-podcast_processed,malaysian-podcast_processed')
@click.option('--replication', default = 1)
def main(folders, replication):
    devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if devices is None:
        devices = list(range(torch.cuda.device_count()))
    else:
        devices = [d.strip() for d in devices.split(',')]

    files = []
    for f in folders.split(','):
        files.extend(glob(f'{f}/**/*.json', recursive = True))
    len(files)

    devices = replication * devices
    print(devices)
    
    df_split = chunks(files, devices)
    pool = Pool(len(devices))
    pooled = pool.map(loop, df_split)
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()