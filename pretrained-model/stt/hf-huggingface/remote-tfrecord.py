import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
import requests
import string
import json
import shutil
import torch
import random
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

CTC_VOCAB = [''] + list(string.ascii_lowercase + string.digits) + [' ']


def download_file_cloud(url, filename):
    r = requests.get(url, stream=True)
    total_size = int(r.headers['content-length'])
    version = int(r.headers.get('X-Bz-Upload-Timestamp', 0))
    try:
        local_size = os.path.getsize(filename)
        if local_size == total_size:
            print(f'{filename} local size matched with cloud size')
            return version
    except Exception as e:
        print(e)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        for data in r.iter_content(chunk_size=1_048_576):
            f.write(data)


def get_dataset(files, directory='tfrecord', overwrite_directory=True):
    if overwrite_directory:
        shutil.rmtree(directory)
    files_to_download = []
    for f in files:
        filename = os.path.join(directory, '-'.join(f.split('/')[-2:]))
        files_to_download.append((f, filename))

    pool = Pool(processes=len(files))
    pool.starmap(download_file_cloud, files_to_download)
    pool.close()
    pool.join()
    tfrecords = glob(f'{directory}/*.tfrecord')
    return tfrecords


def parse(serialized_example):

    data_fields = {
        'waveforms': tf.compat.v1.VarLenFeature(tf.float32),
        'targets': tf.compat.v1.VarLenFeature(tf.int64),
        'targets_length': tf.compat.v1.VarLenFeature(tf.int64),
        'lang': tf.compat.v1.VarLenFeature(tf.int64),
    }
    features = tf.compat.v1.parse_single_example(
        serialized_example, features=data_fields
    )
    for k in features.keys():
        features[k] = features[k].values

    keys = list(features.keys())
    for k in keys:
        if k not in ['waveforms', 'waveforms_length', 'targets']:
            features.pop(k, None)

    return features


class MalayaDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        files,
        directory,
        sr=16000,
        maxlen=16,
        minlen=2,
        batch_files=5,
        max_batch=999999,
        overwrite_directory=True,
        shuffle=True,
        start=True,

    ):
        self.files = [t.replace('gs://mesolitica-tpu-general',
                                'https://huggingface.co/huseinzol05/STT-Mixed-TFRecord/resolve/main') for t in files]
        self.directory = directory
        self.batch_files = batch_files
        self.i = 0
        self.d = None
        self.sr = sr
        self.maxlen = maxlen
        self.minlen = minlen
        self.max_batch = max_batch
        self.overwrite_directory = overwrite_directory
        self.shuffle = shuffle
        if start:
            self.get_dataset()

    def get_dataset(self, num_cpu_threads=4, thread_count=12):
        if self.i >= len(self.files) or self.i == 0:
            self.i = 0
            if self.shuffle:
                random.shuffle(self.files)
        b = self.files[self.i: self.i + self.batch_files]
        tfrecords = get_dataset(b, directory=self.directory, overwrite_directory=self.overwrite_directory)
        d = tf.data.Dataset.from_tensor_slices(tf.constant(tfrecords))
        d = d.shuffle(buffer_size=len(tfrecords))
        cycle_length = min(num_cpu_threads, len(tfrecords))
        d = d.interleave(
            tf.data.TFRecordDataset,
            cycle_length=cycle_length,
            block_length=thread_count)
        d = d.shuffle(buffer_size=100)
        d = d.map(parse, num_parallel_calls=num_cpu_threads)
        d = d.filter(
            lambda x: tf.less(tf.shape(x['waveforms'])[0] / self.sr, self.maxlen)
        )
        d = d.filter(
            lambda x: tf.greater(tf.shape(x['waveforms'])[0] / self.sr, self.minlen)
        )
        self.d = d.as_numpy_iterator()
        self.i += self.batch_files

    def __getitem__(self, idx, raise_exception=False):
        try:
            r = next(self.d)
        except Exception as e:
            if raise_exception:
                raise
            print('Exception __getitem__', e)
            self.get_dataset()
            r = next(self.d)
        r = {'speech': [r['waveforms']], 'sampling_rate': [16000],
             'target_text': ''.join([CTC_VOCAB[t] for t in r['targets']])}
        return r

    def __len__(self):
        return self.max_batch


# https://github.com/huseinzol05/malaya-speech/blob/master/pretrained-model/prepare-stt/3mixed-train-test.json
with open('3mixed-train-test.json') as fopen:
    dataset = json.load(fopen)['train'][:5]

dataset = MalayaDataset(dataset[:5], directory='tfrecord', shuffle=False, overwrite_directory=False)
print(dataset[0])
