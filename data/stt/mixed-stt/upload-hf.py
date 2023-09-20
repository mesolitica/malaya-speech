import json
import requests
from glob import glob
from tqdm import tqdm
import os
from malaya_boilerplate.huggingface import upload_dict
from huggingface_hub import list_repo_files

with open('3mixed-train-test-v2.json') as fopen:
    dataset = json.load(fopen)


def download_file_cloud(url, filename):
    r = requests.get(url, stream=True)
    total_size = int(r.headers['content-length'])
    version = int(r.headers.get('X-Bz-Upload-Timestamp', 0))
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        for data in tqdm(
            iterable=r.iter_content(chunk_size=1_048_576),
            total=total_size / 1_048_576,
            unit='MB',
            unit_scale=True,
        ):
            f.write(data)
    return version


files = [t.replace('gs://mesolitica-tpu-general',
                   'https://f000.backblazeb2.com/file/malay-dataset/speech/mixed') for t in dataset['train']]


def upload(files, directory='tfrecord'):
    files, _ = files
    for f in tqdm(files):
        filename = os.path.join(directory, '-'.join(f.split('/')[-2:]))
        filename_hf = '/'.join(f.split('/')[-2:])
        list_files = list_repo_files('huseinzol05/STT-Mixed-TFRecord')
        if filename_hf in list_files:
            continue
        download_file_cloud(f, filename)
        files_mapping = {filename: filename_hf}
        while True:
            try:
                upload_dict(model='STT-Mixed-TFRecord', files_mapping=files_mapping)
                break
            except Exception as e:
                print(e)

        os.system(f'rm {filename}')


import mp

mp.multiprocessing(files, upload, cores=5, returned=False)
