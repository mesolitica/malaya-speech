import os
import warnings

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
import malaya_speech
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt
import malaya_speech.augmentation.waveform as augmentation
from malaya_speech.train.model import unet as unet
from malaya_speech.utils import tf_featurization
import malaya_speech.train as train
import random
from glob import glob
from collections import defaultdict
from itertools import cycle
from multiprocessing import Pool
import itertools
import pandas as pd


def chunks(l, n):
    for i in range(0, len(l), n):
        yield (l[i : i + n], i // n)


def multiprocessing(strings, function, cores = 6, returned = True):
    df_split = chunks(strings, len(strings) // cores)
    pool = Pool(cores)
    print('initiate pool map')
    pooled = pool.map(function, df_split)
    print('gather from pool')
    pool.close()
    pool.join()
    print('closed pool')

    if returned:
        return list(itertools.chain(*pooled))


librispeech = glob('../speech-bahasa/LibriSpeech/*/*/*/*.flac')
len(librispeech)


def get_speaker_librispeech(file):
    return file.split('/')[-1].split('-')[0]


speakers = defaultdict(list)
for f in librispeech:
    speakers[get_speaker_librispeech(f)].append(f)

len(speakers)

es_ar_male = glob('../speech-bahasa/es_ar_male/*.wav')

speakers_es_ar_male = defaultdict(list)

for f in es_ar_male:
    speakers_es_ar_male[f.split('/')[-1].split('_')[1]].append(f)

len(speakers_es_ar_male)

df_nepali = pd.read_csv(
    '../speech-bahasa/asr_nepali/utt_spk_text.tsv', sep = '\t', header = None
)
asr_nepali = glob('../speech-bahasa/asr_nepali/data/*/*.flac')
asr_nepali.extend(glob('../speech-bahasa/nepali_1/asr_nepali/data/*/*.flac'))
asr_nepali_replaced = {
    f.split('/')[-1].replace('.flac', ''): f for f in asr_nepali
}
df_nepali = df_nepali[df_nepali[0].isin(asr_nepali_replaced.keys())]

speakers_nepali = defaultdict(list)
for i in range(len(df_nepali)):
    speakers_nepali[df_nepali.iloc[i, 1]].append(
        asr_nepali_replaced[df_nepali.iloc[i, 0]]
    )

len(speakers_nepali)

df_sinhala = pd.read_csv(
    '../speech-bahasa/sinhala_0/asr_sinhala/utt_spk_text.tsv',
    sep = '\t',
    header = None,
)
asr_sinhala = glob('../speech-bahasa/sinhala_0/asr_sinhala/data/*/*.flac')
asr_sinhala.extend(glob('../speech-bahasa/sinhala_1/asr_sinhala/data/*/*.flac'))

asr_sinhala_replaced = {
    f.split('/')[-1].replace('.flac', ''): f for f in asr_sinhala
}
df_sinhala = df_sinhala[df_sinhala[0].isin(asr_sinhala_replaced.keys())]

speakers_sinhala = defaultdict(list)
for i in range(len(df_sinhala)):
    speakers_sinhala[df_sinhala.iloc[i, 1]].append(
        asr_sinhala_replaced[df_sinhala.iloc[i, 0]]
    )

len(speakers_sinhala)

files = glob('../speech-bahasa/ST-CMDS-20170001_1-OS/*.wav')
speakers_mandarin = defaultdict(list)
for f in files:
    speakers_mandarin[f[:-9]].append(f)
len(speakers_mandarin)

speakers_malay = {}
speakers_malay['salina'] = glob(
    '../youtube/malay2/salina/output-wav-salina/*.wav'
)
speakers_malay['turki'] = glob('../youtube/malay2/turki/output-wav-turki/*.wav')
speakers_malay['dari-pasentran-ke-istana'] = glob(
    '../youtube/malay/dari-pasentran-ke-istana/output-wav-dari-pasentran-ke-istana/*.wav'
)

noises = glob('../noise-44k/noise/*.wav') + glob('../noise-44k/clean-wav/*.wav')
basses = glob('HHDS/Sources/**/*bass.wav', recursive = True)
drums = glob('HHDS/Sources/**/*drums.wav', recursive = True)
others = glob('HHDS/Sources/**/*other.wav', recursive = True)
noises = noises + basses + drums + others
random.shuffle(noises)
sr = 8000
speakers_size = 4

s = {
    **speakers,
    **speakers_es_ar_male,
    **speakers_nepali,
    **speakers_sinhala,
    **speakers_mandarin,
    **speakers_malay,
}

keys = list(s.keys())


def random_speakers(n):
    ks = random.sample(keys, n)
    r = []
    for k in ks:
        r.append(random.choice(s[k]))
    return r


def read_wav(f):
    return malaya_speech.load(f, sr = sr)


def random_sampling(s, length):
    return augmentation.random_sampling(s, sr = sr, length = length)


def combine_speakers(files, n = 5, limit = 4):
    w_samples = random.sample(files, n)
    w_samples = [
        random_sampling(
            read_wav(f)[0],
            length = min(random.randint(10000 // n, 20000 // n), 10000),
        )
        for f in w_samples
    ]
    y = [w_samples[0]]
    left = w_samples[0].copy() * random.uniform(0.5, 1.0)

    combined = None

    for i in range(1, n):
        right = w_samples[i].copy() * random.uniform(0.5, 1.0)
        overlap = random.uniform(0.1, 0.9)
        print(i, overlap)
        len_overlap = int(overlap * len(right))
        minus = len(left) - len_overlap
        padded_right = np.pad(right, (minus, 0))
        left = np.pad(left, (0, len(padded_right) - len(left)))

        left = left + padded_right

        if i >= (limit - 1):
            if combined is None:
                combined = padded_right
            else:
                combined = np.pad(
                    combined, (0, len(padded_right) - len(combined))
                )
                combined += padded_right

        else:
            y.append(padded_right)

    if combined is not None:
        y.append(combined)

    for i in range(len(y)):
        if len(y[i]) != len(left):
            y[i] = np.pad(y[i], (0, len(left) - len(y[i])))
            y[i] = y[i] / np.max(np.abs(y[i]))

    left = left / np.max(np.abs(left))
    return left, y


def parallel(f):
    count = random.randint(0, 10)
    while True:
        try:
            if count > 0:
                combined, y = combine_speakers(random_speakers(count), count)
            else:
                combined, y = combine_speakers(noises, random.randint(1, 10))
                for i in range(len(y)):
                    y[i] = np.zeros((len(combined)))

            if 10 < (len(combined) / sr):
                break
        except:
            pass

    while len(y) < 4:
        y.append(np.zeros((len(combined))))

    y = np.array(y)
    return combined, y


def loop(files):
    files = files[0]
    results = []
    for f in files:
        results.append(parallel(f))
    return results


def generate(batch_size = 10, repeat = 20):
    fs = [i for i in range(batch_size)]
    while True:
        results = multiprocessing(fs, loop, cores = len(fs))
        for _ in range(repeat):
            random.shuffle(results)
            for r in results:
                if not np.isnan(r[0]).any() and not np.isnan(r[1]).any():
                    yield {'combined': r[0], 'y': r[1]}


def get_dataset(batch_size = 2):
    def get():
        dataset = tf.data.Dataset.from_generator(
            generate,
            {'combined': tf.float32, 'y': tf.float32},
            output_shapes = {
                'combined': tf.TensorShape([None]),
                'y': tf.TensorShape([speakers_size, None]),
            },
        )
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes = {
                'combined': tf.TensorShape([None]),
                'y': tf.TensorShape([speakers_size, None]),
            },
            padding_values = {
                'combined': tf.constant(0, dtype = tf.float32),
                'y': tf.constant(0, dtype = tf.float32),
            },
        )
        return dataset

    return get
