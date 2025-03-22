import os
import librosa
import json
import click
import torch
from multiprocess import Pool
from tqdm import tqdm
from datasets import load_dataset

def new_path(f):
    f = f.replace('.mp3', '.snac')
    splitted = f.split('/')
    base_folder = splitted[0] + '_snac'
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

def tokenise_audio(snac_model, waveform):
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    waveform = waveform.to(dtype=torch.float32)
    waveform = waveform.unsqueeze(0)
    with torch.inference_mode():
        codes = snac_model.encode(waveform.to('cuda'))
    for i in range(len(codes)):
        codes[i] = codes[i].cpu().numpy()

    all_codes = []
    for i in range(codes[0].shape[1]):
        all_codes.append(codes[0][0][i]+128266)
        all_codes.append(codes[1][0][2*i]+128266+4096)
        all_codes.append(codes[2][0][4*i]+128266+(2*4096))
        all_codes.append(codes[2][0][(4*i)+1]+128266+(3*4096))
        all_codes.append(codes[1][0][(2*i)+1]+128266+(4*4096))
        all_codes.append(codes[2][0][(4*i)+2]+128266+(5*4096))
        all_codes.append(codes[2][0][(4*i)+3]+128266+(6*4096))

    return all_codes

def loop(rows):
    rows, index = rows
    os.environ['CUDA_VISIBLE_DEVICES'] = str(index)
    
    import torch
    from snac import SNAC
    
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    snac_model = snac_model.eval().cuda()

    for row in tqdm(rows):
        f = row['audio_filename']
        splitted = new_path(f)
        if os.path.exists(splitted):
            continue
            
        y, _ = librosa.load(f, sr = 24000)
        myts = tokenise_audio(y)
        myts = [int(i) for i in myts]
        
        os.makedirs(os.path.split(splitted)[0], exist_ok = True)
        with open(splitted, 'w') as fopen:
            json.dump(myts, fopen)

@click.command()
@click.option('--dataset', default = 'mesolitica/TTS-Combined')
@click.option('--split', default = 'train')
@click.option('--replication', default = 1)
def main(dataset, split, replication):
    devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if devices is None:
        devices = list(range(torch.cuda.device_count()))
    else:
        devices = [d.strip() for d in devices.split(',')]

    devices = replication * devices
    print(devices)

    dataset = load_dataset(dataset)[split]
    rows = dataset.to_list()

    df_split = chunks(rows, devices)
    pool = Pool(len(devices))
    pooled = pool.map(loop, df_split)
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()

