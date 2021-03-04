import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
import malaya_speech.train as train
import malaya_speech.train.model.vggvox_v2 as vggvox_v2
import malaya_speech.augmentation.waveform as augmentation
import malaya_speech
from glob import glob
import librosa
import numpy as np
from collections import defaultdict
from itertools import cycle
from multiprocessing import Pool
import itertools
import pandas as pd
import random


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

vctk = glob('vtck/**/*.flac', recursive = True)
vctk_speakers = defaultdict(list)
for f in vctk:
    s = f.split('/')[-1].split('_')[0]
    vctk_speakers[s].append(f)

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

noises = glob('/home/husein/youtube/noise-22k/*.wav')
random.shuffle(noises)
sr = 16000

s = {**speakers, **vctk_speakers, **speakers_mandarin, **speakers_malay}

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


def combine_speakers(files, n = 5):
    w_samples = random.sample(files, n)
    w_samples = [
        random_sampling(
            read_wav(f)[0], length = random.randint(500, max(10000 // n, 5000))
        )
        for f in w_samples
    ]
    left = w_samples[0].copy() * random.uniform(0.5, 1.0)

    combined = None

    for i in range(1, n):
        right = w_samples[i].copy() * random.uniform(0.5, 1.0)
        overlap = random.uniform(0.1, 0.9)
        len_overlap = int(overlap * len(right))
        minus = len(left) - len_overlap
        if minus < 0:
            minus = 0
        padded_right = np.pad(right, (minus, 0))
        left = np.pad(left, (0, len(padded_right) - len(left)))

        left = left + padded_right

    left = left / np.max(np.abs(left))
    return left


labels = [
    '0 speaker',
    '1 speaker',
    '2 speakers',
    '3 speakers',
    '4 speakers',
    '5 speakers',
    '6 speakers',
    '7 speakers',
    '8 speakers',
    '9 speakers',
    '10 speakers',
    '11 speakers',
    '12 speakers',
    'more than 12 speakers',
]


def parallel(f):
    count = random.randint(0, 15)
    if count > 12:
        count = random.randint(13, 20)
    while True:
        try:
            if count > 0:
                combined = combine_speakers(random_speakers(count), count)
            else:
                combined = combine_speakers(noises, random.randint(1, 10))
            break
        except Exception as e:
            print(e)
            pass
    if count > (len(labels) - 1):
        print(count)
        count = len(labels) - 1

    return combined, [count]


def loop(files):
    files = files[0]
    results = []
    for f in files:
        results.append(parallel(f))
    return results


def generate(batch_size = 10, repeat = 100):
    fs = [i for i in range(batch_size)]
    while True:
        results = multiprocessing(fs, loop, cores = len(fs))
        for _ in range(repeat):
            random.shuffle(results)
            for r in results:
                if not np.isnan(r[0]).any() and not np.isnan(r[1]).any():
                    yield {'inputs': r[0], 'targets': r[1]}


def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft = 1024):
    linear = librosa.stft(
        wav, n_fft = n_fft, win_length = win_length, hop_length = hop_length
    )  # linear spectrogram
    return linear.T


def load_data(
    wav,
    win_length = 400,
    sr = 16000,
    hop_length = 300,
    n_fft = 512,
    spec_len = 250,
    mode = 'train',
):
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    if mode == 'train':
        if time > spec_len:
            randtime = np.random.randint(0, time - spec_len)
            spec_mag = mag_T[:, randtime : randtime + spec_len]
        else:
            spec_mag = np.pad(mag_T, ((0, 0), (0, spec_len - time)), 'constant')
    else:
        spec_mag = mag_T
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims = True)
    std = np.std(spec_mag, 0, keepdims = True)
    return (spec_mag - mu) / (std + 1e-5)


DIMENSION = 257


def calc(v):

    r = load_data(v, mode = 'eval')
    return r


def preprocess_inputs(example):
    s = tf.compat.v1.numpy_function(calc, [example['inputs']], tf.float32)

    s = tf.reshape(s, (DIMENSION, -1, 1))
    example['inputs'] = s

    return example


def get_dataset(batch_size = 16, shuffle_size = 256, thread_count = 6):
    def get():
        dataset = tf.data.Dataset.from_generator(
            generate,
            {'inputs': tf.float32, 'targets': tf.int32},
            output_shapes = {
                'inputs': tf.TensorShape([None]),
                'targets': tf.TensorShape([None]),
            },
        )
        dataset = dataset.map(preprocess_inputs)
        dataset = dataset.shuffle(shuffle_size)
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes = {
                'inputs': tf.TensorShape([DIMENSION, None, 1]),
                'targets': tf.TensorShape([None]),
            },
            padding_values = {
                'inputs': tf.constant(0, dtype = tf.float32),
                'targets': tf.constant(0, dtype = tf.int32),
            },
        )
        return dataset

    return get


learning_rate = 1e-5
init_checkpoint = '../vggvox-speaker-identification/v2/vggvox.ckpt'


def model_fn(features, labels, mode, params):
    Y = tf.cast(features['targets'][:, 0], tf.int32)
    model = vggvox_v2.Model(features['inputs'], num_class = 14, mode = 'train')

    logits = model.logits
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = logits, labels = Y
        )
    )

    tf.identity(loss, 'train_loss')

    accuracy = tf.metrics.accuracy(
        labels = Y, predictions = tf.argmax(logits, axis = 1)
    )

    tf.identity(accuracy[1], name = 'train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    variables = [v for v in variables if 'prediction' not in v.name]

    assignment_map, initialized_variable_names = train.get_assignment_map_from_checkpoint(
        variables, init_checkpoint
    )

    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step = global_step)
        estimator_spec = tf.estimator.EstimatorSpec(
            mode = mode, loss = loss, train_op = train_op
        )

    elif mode == tf.estimator.ModeKeys.EVAL:

        estimator_spec = tf.estimator.EstimatorSpec(
            mode = tf.estimator.ModeKeys.EVAL,
            loss = loss,
            eval_metric_ops = {'accuracy': accuracy},
        )

    return estimator_spec


train_hooks = [
    tf.train.LoggingTensorHook(
        ['train_accuracy', 'train_loss'], every_n_iter = 1
    )
]

train_dataset = get_dataset()

save_directory = 'output-vggvox-v2-speaker-count'

train.run_training(
    train_fn = train_dataset,
    model_fn = model_fn,
    model_dir = save_directory,
    num_gpus = 1,
    log_step = 1,
    save_checkpoint_step = 25000,
    max_steps = 300000,
    train_hooks = train_hooks,
)
