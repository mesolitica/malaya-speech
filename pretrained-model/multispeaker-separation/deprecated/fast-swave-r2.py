import os
import warnings

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
warnings.filterwarnings('ignore')

import tensorflow as tf
import malaya_speech
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt
import malaya_speech.augmentation.waveform as augmentation
from malaya_speech.train.model import fast_swave, fastspeech
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
        yield (l[i: i + n], i // n)


def multiprocessing(strings, function, cores=6, returned=True):
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

vctk = glob('vtck/**/*.flac', recursive=True)
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

noises = glob('../noise-44k/noise/*.wav') + glob('../noise-44k/clean-wav/*.wav')
basses = glob('HHDS/Sources/**/*bass.wav', recursive=True)
drums = glob('HHDS/Sources/**/*drums.wav', recursive=True)
others = glob('HHDS/Sources/**/*other.wav', recursive=True)
noises = noises + basses + drums + others
random.shuffle(noises)
sr = 22050
speakers_size = 4

s = {**speakers, **vctk_speakers, **speakers_mandarin, **speakers_malay}

keys = list(s.keys())


def random_speakers(n):
    ks = random.sample(keys, n)
    r = []
    for k in ks:
        r.append(random.choice(s[k]))
    return r


def read_wav(f):
    return malaya_speech.load(f, sr=sr)


def random_sampling(s, length):
    return augmentation.random_sampling(s, sr=sr, length=length)


def combine_speakers(files, n=5, limit=4):
    w_samples = random.sample(files, n)
    w_samples = [
        random_sampling(
            read_wav(f)[0], length=random.randint(1500, max(10000 // n, 6000))
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
        if minus < 0:
            minus = 0
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
    count = random.randint(0, 5)
    while True:
        try:
            if count > 0:
                combined, y = combine_speakers(random_speakers(count), count)
            else:
                combined, y = combine_speakers(noises, random.randint(1, 5))
                for i in range(len(y)):
                    y[i] = np.zeros((len(combined)))

            while len(y) < 4:
                y.append(np.zeros((len(combined))))

            y = np.array(y)
            print('len', len(combined) / sr)

            combined = malaya_speech.featurization.universal_mel(combined)
            y = [malaya_speech.featurization.universal_mel(i) for i in y]

            break
        except BaseException:
            pass

    return combined, y, [len(combined)]


def loop(files):
    files = files[0]
    results = []
    for f in files:
        results.append(parallel(f))
    return results


def generate(batch_size=10, repeat=50):
    fs = [i for i in range(batch_size)]
    while True:
        results = multiprocessing(fs, loop, cores=len(fs))
        for _ in range(repeat):
            random.shuffle(results)
            for r in results:
                if not np.isnan(r[0]).any() and not np.isnan(r[1]).any():
                    yield {'combined': r[0], 'y': r[1], 'length': r[2]}


def get_dataset(batch_size=4):
    def get():
        dataset = tf.data.Dataset.from_generator(
            generate,
            {'combined': tf.float32, 'y': tf.float32, 'length': tf.int32},
            output_shapes={
                'combined': tf.TensorShape([None, 80]),
                'y': tf.TensorShape([speakers_size, None, 80]),
                'length': tf.TensorShape([None]),
            },
        )
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes={
                'combined': tf.TensorShape([None, 80]),
                'y': tf.TensorShape([speakers_size, None, 80]),
                'length': tf.TensorShape([None]),
            },
            padding_values={
                'combined': tf.constant(0, dtype=tf.float32),
                'y': tf.constant(0, dtype=tf.float32),
                'length': tf.constant(0, dtype=tf.int32),
            },
        )
        return dataset

    return get


total_steps = 300000


def model_fn(features, labels, mode, params):
    lengths = features['length'][:, 0]
    config = malaya_speech.config.fastspeech_config
    dim = 192
    config['encoder_hidden_size'] = dim
    config['decoder_hidden_size'] = dim * speakers_size
    config = fastspeech.Config(vocab_size=1, **config)
    model = fast_swave.Model(config, R=2, C=speakers_size, N=dim, O=dim)
    outputs, output_all = model(features['combined'], lengths)

    loss = 0
    for c_idx, est_src in enumerate(outputs):
        coeff = (c_idx + 1) * (1 / len(outputs))
        abs_loss = fast_swave.calculate_loss(
            features['y'], est_src, lengths, C=speakers_size
        )
        loss += abs_loss

    loss /= len(outputs)
    tf.identity(loss, 'total_loss')
    tf.summary.scalar('total_loss', loss)

    global_step = tf.train.get_or_create_global_step()

    if mode == tf.estimator.ModeKeys.TRAIN:

        train_op = train.optimizer.adamw.create_optimizer(
            loss,
            init_lr=0.0001,
            num_train_steps=total_steps,
            num_warmup_steps=100000,
            end_learning_rate=0.00005,
            weight_decay_rate=0.001,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-6,
            clip_norm=1.0,
        )
        estimator_spec = tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op
        )

    elif mode == tf.estimator.ModeKeys.EVAL:

        estimator_spec = tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL, loss=loss
        )

    return estimator_spec


train_hooks = [tf.train.LoggingTensorHook(['total_loss'], every_n_iter=1)]
train_dataset = get_dataset()

save_directory = 'speaker-split-fast-swave-r2'

train.run_training(
    train_fn=train_dataset,
    model_fn=model_fn,
    model_dir=save_directory,
    num_gpus=1,
    log_step=1,
    save_checkpoint_step=3000,
    max_steps=total_steps,
    train_hooks=train_hooks,
    eval_step=0,
)
