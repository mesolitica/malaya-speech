import os
import json
import click
import torch
import math
import librosa
from glob import glob
from multiprocess import Pool
from tqdm import tqdm

def new_path(f):
    f = f.replace('.mp3', '.25tps')
    splitted = f.split('/')
    base_folder = splitted[0] + '_25tps'
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
    torch.set_grad_enabled(False)

    from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer
    from torch.utils.data import Dataset, DataLoader

    model_id = "mesolitica/whisper-25TPS-VQ-32k-large-v3-turbo"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        torch_dtype=torch.float16,
    ).cuda()
    encoder = model.model.get_encoder()

    class CustomDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows
    
        def __len__(self):
            return len(self.rows)
    
        def __getitem__(self, idx):
            try:
                return self.rows[idx], librosa.load(self.rows[idx], sr = 16000)[0]
            except Exception as e:
                print('error in dataset', e, self.rows[idx])
                return None
    
    def collator(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        filenames = [b[0] for b in batch]
        audio = [b[1] for b in batch]
        return {'filenames': filenames, 'audio': audio}

    data = CustomDataset(rows)
    dataloader = DataLoader(
        data, 
        batch_size=24, 
        collate_fn=collator, 
        num_workers=10, 
        prefetch_factor=5, 
        pin_memory=True,
    )
    
    for batch in tqdm(iter(dataloader)):
        if batch is None:
            continue
        try:
            utts = batch['audio']
            audios, indices = [], []
            for idx, utt in enumerate(utts):
                audio = utt
                time_step = 0
                while time_step * 16000 < audio.shape[0]:
                    audio_segment = audio[time_step * 16000: (time_step + 30) * 16000]
                    audios.append(audio_segment)
                    indices.append(idx)
                    time_step += 30

            all_speech_tokens = [[] for _ in range(len(utts))]
            batch_size = 128
            for start in range(0, len(audios), batch_size):
                features = feature_extractor(
                    audios[start: start + batch_size], 
                    return_tensors = 'pt', 
                    return_attention_mask = True,
                    sampling_rate = 16000,
                )
                features['input_features'] = features['input_features'].to(torch.float16).cuda()
                features['attention_mask'] = features['attention_mask'].cuda()
                encoded = encoder(**features)
                tokens = encoded[1].cpu()
                attention_mask = encoded[2].cpu()
                for i in range(len(tokens)):
                    idx = indices[start + i]
                    e = tokens[i, attention_mask[i] == 1].tolist()
                    all_speech_tokens[idx].extend(e)

            for no, f in enumerate(batch['filenames']):
                e = all_speech_tokens[no]
                splitted = new_path(f)
                os.makedirs(os.path.split(splitted)[0], exist_ok = True)
                with open(splitted, 'w') as fopen:
                    json.dump(e, fopen)

        except Exception as e:
            print(e)

@click.command()
@click.option('--path', default = 'science-segment/*.mp3')
@click.option('--replication', default = 1)
def main(path, replication):
    devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if devices is None:
        devices = list(range(torch.cuda.device_count()))
    else:
        devices = [d.strip() for d in devices.split(',')]

    devices = replication * devices
    print(devices)
    
    rows = glob(path)
    print(len(rows))
    rows = [f for f in tqdm(rows) if not os.path.exists(new_path(f))]
    print(len(rows))
    
    df_split = chunks(rows, devices)
    pool = Pool(len(devices))
    pooled = pool.map(loop, df_split)
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()