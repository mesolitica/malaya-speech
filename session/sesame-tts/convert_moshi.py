import os
import librosa
import json
import click
import torch
from glob import glob
from multiprocess import Pool
from tqdm import tqdm

def new_path(f):
    f = f.replace('.mp3', '.moshi')
    splitted = f.split('/')
    base_folder = splitted[0] + '_moshi'
    splitted = '/'.join([base_folder] + splitted[1:])
    return splitted

def chunks(l, devices):
    chunk_size = len(l) // len(devices)
    remainder = len(l) % len(devices)
    start = 0
    for i in range(len(devices)):
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        yield (l[start:end], devices[i])
        start = end

def loop(rows):
    rows, index = rows
    os.environ['CUDA_VISIBLE_DEVICES'] = str(index)
    
    import torch
    import torchaudio
    from moshi.models import loaders
    from huggingface_hub import hf_hub_download
    
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device='cuda')
    mimi.set_num_codebooks(32)

    for f in tqdm(rows):
        splitted = new_path(f)
        if os.path.exists(splitted):
            continue
        
        audio_tensor, sample_rate = torchaudio.load(f)
        if audio_tensor.shape[0] != 1:
            audio_tensor = audio_tensor.mean(dim=0)
        audio_tensor = audio_tensor.squeeze(0)
        if sample_rate != mimi._sample_rate:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, orig_freq=sample_rate, new_freq=generator.sample_rate
            )
        
        tokens = mimi.encode(audio_tensor.to('cuda').unsqueeze(0).unsqueeze(0))[0].cpu().numpy()
        os.makedirs(os.path.split(splitted)[0], exist_ok = True)
        with open(splitted, 'w') as fopen:
            json.dump(tokens.tolist(), fopen)

@click.command()
@click.option('--path', default = '*_trim/**/*.mp3')
@click.option('--replication', default = 1)
def main(path, replication):
    devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if devices is None:
        devices = list(range(torch.cuda.device_count()))
    else:
        devices = [d.strip() for d in devices.split(',')]

    devices = replication * devices
    print(devices)
    
    rows = glob(path, recursive = True)
    rows = [f for f in rows if not os.path.exists(new_path(f))]
    
    df_split = chunks(rows, devices)
    pool = Pool(len(devices))
    pooled = pool.map(loop, df_split)
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()