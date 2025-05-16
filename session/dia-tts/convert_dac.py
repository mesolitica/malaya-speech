import os
import json
import click
import torch
import math
from glob import glob
from multiprocess import Pool
from tqdm import tqdm

def new_path(f):
    f = f.replace('.mp3', '.dac')
    splitted = f.split('/')
    base_folder = splitted[0] + '_dac'
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
    from torch.utils.data import Dataset, DataLoader
    from transformers import DacModel, AutoProcessor
    
    model = DacModel.from_pretrained("descript/dac_44khz")
    model.cuda()
    processor = AutoProcessor.from_pretrained("descript/dac_44khz")

    class CustomDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            try:
                audio_tensor, sample_rate = torchaudio.load(self.rows[idx])
                if len(audio_tensor.shape) > 1:
                    audio_tensor = audio_tensor.mean(dim=0)
                audio_tensor = torchaudio.functional.resample(audio_tensor, sample_rate, processor.sampling_rate)

                length = math.ceil(len(audio_tensor) / model.config.hop_length)
                inputs = processor(
                    raw_audio=audio_tensor, 
                    sampling_rate=processor.sampling_rate, 
                    return_tensors="pt",
                )
                return {'audio_tensor': inputs['input_values'][0,0], 'filename': self.rows[idx], 'length': length}
            except Exception as e:
                print(e)

    def collator(batch):
        batch = [b for b in batch if b is not None]
        audio_tensors = [b['audio_tensor'] for b in batch]
        filenames = [b['filename'] for b in batch]
        lengths = [b['length'] for b in batch]
        padded = torch.nn.utils.rnn.pad_sequence(audio_tensors, batch_first = True)
        return {'padded': padded, 'filenames': filenames, 'lengths': lengths}

    data = CustomDataset(rows)
    dataloader = DataLoader(data, batch_size=24, collate_fn = collator, 
                       num_workers = 10, prefetch_factor = 5, pin_memory = True)

    with torch.no_grad():
        for batch in tqdm(iter(dataloader)):
            padded = batch['padded'].cuda()
            lengths = batch['lengths']
            filenames = batch['filenames']
            encoder_outputs = model.encode(padded.unsqueeze(1))
            audio_codes = encoder_outputs.audio_codes.cpu().numpy()
            for i in range(len(lengths)):
                f = filenames[i]
                splitted = new_path(f)
                os.makedirs(os.path.split(splitted)[0], exist_ok = True)
                tokens = audio_codes[i, :, :lengths[i]]
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