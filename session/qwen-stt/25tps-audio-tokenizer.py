import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import json
import click
import os
import librosa
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer
import librosa
import torch

torch.autograd.set_grad_enabled(False)

def new_path(f):
    f = f.replace('.mp3', '.25tps')
    splitted = f.split('/')
    base_folder = splitted[0] + '_25tps'
    splitted = '/'.join([base_folder] + splitted[1:])
    return splitted

@click.command()
@click.option('--path', default = 'science-segment/*.mp3')
@click.option('--batch_size', default = 400)
def main(path, batch_size):

    model_id = "mesolitica/whisper-25TPS-VQ-32k-large-v3-turbo"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code = True, torch_dtype = torch.float16).cuda()
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
        filenames = [b[0] for b in batch]
        audio = [b[1] for b in batch]
        features = feature_extractor(
            audio, 
            return_tensors = 'pt', 
            return_attention_mask = True,
            sampling_rate = 16000,
        )
        return {'filenames': filenames, 'features': features}
    
    rows = glob(path)
    rows = [f for f in tqdm(rows) if not os.path.exists(new_path(f))]
    data = CustomDataset(rows)
    dataloader = DataLoader(
        data, 
        batch_size=batch_size, 
        collate_fn=collator, 
        num_workers=10, 
        prefetch_factor=5,
        pin_memory=True,
    )

    for batch in tqdm(iter(dataloader)):
        try:
            features = batch['features']
            features['input_features'] = features['input_features'].to(torch.float16).cuda()
            features['attention_mask'] = features['attention_mask'].cuda()
            encoded = encoder(**features)
            tokens = encoded[1].cpu()
            attention_mask = encoded[2].cpu()

            for no, f in enumerate(batch['filenames']):
                e = tokens[no, attention_mask[no] == 1].tolist()
                splitted = new_path(f)
                os.makedirs(os.path.split(splitted)[0], exist_ok = True)
                with open(splitted, 'w') as fopen:
                    json.dump(e, fopen)

        except Exception as e:
            print(e)

if __name__ == '__main__':
    main()