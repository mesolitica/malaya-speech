import os
import warnings

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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


def get_speaker_librispeech(file):
    return file.split('/')[-1].split('-')[0]


speakers = defaultdict(list)
for f in librispeech:
    speakers[get_speaker_librispeech(f)].append(f)

voxceleb = glob('../voxceleb/wav/**/*/*/*.wav', recursive=True)


def get_speaker_voxceleb(file):
    return file.split('/')[-3]


voxceleb_speakers = defaultdict(list)
for f in voxceleb:
    voxceleb_speakers[get_speaker_voxceleb(f)].append(f)

s = {**voxceleb_speakers, **speakers}

keys = list(s.keys())

noises = glob('../noise-44k/noise/*.wav') + glob('../noise-44k/clean-wav/*.wav')
basses = glob('HHDS/Sources/**/*bass.wav', recursive=True)
drums = glob('HHDS/Sources/**/*drums.wav', recursive=True)
others = glob('HHDS/Sources/**/*other.wav', recursive=True)
noises = noises + basses + drums + others
random.shuffle(noises)

sr = 44100
speakers_size = 4


def random_speakers(n):
    ks = random.sample(keys, n)
    r = []
    for k in ks:
        r.append(random.choice(s[k]))
    return r


def read_wav(f):
    return malaya_speech.load(f, sr=44100)


def random_sampling(s, length):
    return augmentation.random_sampling(s, sr=44100, length=length)


def combine_speakers(files, n=5, limit=4):
    w_samples = random.sample(files, n)
    w_samples = [
        random_sampling(
            read_wav(f)[0],
            length=min(random.randint(10000 // n, 100_000 // n), 12000),
        )
        for f in w_samples
    ]
    y = [w_samples[0]]
    left = w_samples[0].copy() * random.uniform(0.5, 1.0)
    last_right = w_samples[0]

    combined = None

    for i in range(1, n):

        right = w_samples[i].copy() * random.uniform(0.5, 1.0)

        overlap = random.uniform(0.1, 1.2)
        left_len = int(overlap * len(last_right))

        padded_right = np.pad(right, (left_len, 0))

        if len(left) > len(padded_right):
            padded_right = np.pad(
                padded_right, (0, len(left) - len(padded_right))
            )
        else:
            left = np.pad(left, (0, len(padded_right) - len(left)))

        last_right = padded_right

        left = left + padded_right

        if i >= (limit - 1):
            if combined is None:
                combined = padded_right
            else:
                combined = np.pad(
                    combined, (0, len(padded_right) - len(combined))
                )
                combined += padded_right
                print(i, combined.shape, padded_right.shape)

        else:
            y.append(padded_right)

        print(i, padded_right.shape)

    if combined is not None:
        y.append(combined)

    for i in range(len(y)):
        if len(y[i]) != len(left):
            y[i] = np.pad(y[i], (0, len(left) - len(y[i])))

    left = left / np.max(np.abs(left))
    return left, y


def parallel(f):
    count = random.randint(0, 10)
    while True:
        if count > 0:
            combined, y = combine_speakers(random_speakers(count), count)
        else:
            combined, y = combine_speakers(noises, random.randint(1, 10))
            for i in range(len(y)):
                y[i] = np.zeros((len(combined)))

        if (len(combined) / sr) < 30:
            break

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


def generate(batch_size=10, repeat=20):
    fs = [i for i in range(batch_size)]
    while True:
        results = multiprocessing(fs, loop, cores=len(fs))
        for _ in range(repeat):
            random.shuffle(results)
            for r in results:
                if not np.isnan(r[0]).any() and not np.isnan(r[1]).any():
                    yield {'combined': r[0], 'y': r[1]}


def get_dataset():
    def get():
        dataset = tf.data.Dataset.from_generator(
            generate,
            {'combined': tf.float32, 'y': tf.float32},
            output_shapes={
                'combined': tf.TensorShape([None]),
                'y': tf.TensorShape([speakers_size, None]),
            },
        )
        return dataset

    return get


class Model:
    def __init__(self, X, Y, speakers, dropout=0.5, training=True):
        self.X = X
        self.Y = Y
        self.speakers = speakers

        stft_X, D_X = tf_featurization.get_stft(self.X, T=512 * 3)

        self.stft = []
        for i in range(speakers):
            self.stft.append(tf_featurization.get_stft(self.Y[i], T=512 * 3))

        params = {'conv_n_filters': [22 * (2 ** i) for i in range(6)]}

        self.outputs = []
        for i in range(speakers):
            with tf.variable_scope(f'model_{i}'):
                self.outputs.append(
                    unet.Model(
                        D_X,
                        dropout=dropout,
                        training=training,
                        params=params,
                    ).logits
                )

        self.loss = []
        for i in range(speakers):
            self.loss.append(
                tf.reduce_mean(tf.abs(self.outputs[i] - self.stft[i][1]))
            )

        self.cost = tf.reduce_sum(self.loss)


init_lr = 1e-4
epochs = 500_000


def model_fn(features, labels, mode, params):
    model = Model(features['combined'], features['y'], speakers=speakers_size)
    loss = model.cost
    tf.identity(loss, 'total_loss')
    tf.summary.scalar('total_loss', loss)
    for i in range(len(model.loss)):
        l = tf.reduce_mean(model.loss[i])
        tf.identity(l, f'loss_{i}')
        tf.summary.scalar(f'loss_{i}', l)

    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        epochs,
        end_learning_rate=1e-6,
        power=1.0,
        cycle=False,
    )
    tf.summary.scalar('learning_rate', learning_rate)

    if mode == tf.estimator.ModeKeys.TRAIN:

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        train_op = optimizer.minimize(loss, global_step=global_step)
        estimator_spec = tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op
        )

    elif mode == tf.estimator.ModeKeys.EVAL:

        estimator_spec = tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL, loss=loss
        )

    return estimator_spec


train_hooks = [
    tf.train.LoggingTensorHook(
        ['total_loss', 'loss_0', 'loss_1', 'loss_2', 'loss_3'], every_n_iter=1
    )
]
train_dataset = get_dataset()

save_directory = 'speaker-split-unet'

train.run_training(
    train_fn=train_dataset,
    model_fn=model_fn,
    model_dir=save_directory,
    num_gpus=1,
    log_step=1,
    save_checkpoint_step=3000,
    max_steps=epochs,
    train_hooks=train_hooks,
    eval_step=0,
)
