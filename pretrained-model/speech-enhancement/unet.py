import os
import warnings

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
warnings.filterwarnings('ignore')

import tensorflow as tf
import malaya_speech
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt
import malaya_speech.augmentation.waveform as augmentation
from malaya_speech.train.model import unet
from malaya_speech.utils import tf_featurization
import malaya_speech.train as train
import random
from glob import glob
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


files = glob('../youtube/clean-wav/*.wav')
random.shuffle(files)
len(files)

noises = glob('../noise-44k/noise/*.wav') + glob('../noise-44k/clean-wav/*.wav')
basses = glob('HHDS/Sources/**/*bass.wav', recursive=True)
drums = glob('HHDS/Sources/**/*drums.wav', recursive=True)
others = glob('HHDS/Sources/**/*other.wav', recursive=True)
noises = noises + basses + drums + others
random.shuffle(noises)
file_cycle = cycle(files)
len(noises)


def read_wav(f):
    return malaya_speech.load(f, sr=44100)


def random_sampling(s, length):
    return augmentation.random_sampling(s, sr=44100, length=length)


def combine_speakers(files, n=5):
    w_samples = random.sample(files, n)
    w_samples = [
        random_sampling(
            read_wav(f)[0],
            length=min(
                random.randint(20000 // n, 240_000 // n), 100_000 // n
            ),
        )
        for f in w_samples
    ]
    y = [w_samples[0]]
    left = w_samples[0].copy() * random.uniform(0.5, 1.0)
    for i in range(1, n):

        right = w_samples[i].copy() * random.uniform(0.5, 1.0)

        overlap = random.uniform(0.01, 1.25)
        left_len = int(overlap * len(left))

        padded_right = np.pad(right, (left_len, 0))

        if len(left) > len(padded_right):
            padded_right = np.pad(
                padded_right, (0, len(left) - len(padded_right))
            )
        else:
            left = np.pad(left, (0, len(padded_right) - len(left)))

        y.append(padded_right)
        left = left + padded_right
    return left, y


def random_amplitude(sample, low=3, high=5):
    y_aug = sample.copy()
    dyn_change = np.random.uniform(low=low, high=high)
    y_aug = y_aug * dyn_change
    return np.clip(y_aug, -1, 1)


def random_amplitude_threshold(sample, low=1, high=2, threshold=0.4):
    y_aug = sample.copy()
    y_aug = y_aug / (np.max(np.abs(y_aug)) + 1e-9)
    dyn_change = np.random.uniform(low=low, high=high)
    y_aug[np.abs(y_aug) >= threshold] = (
        y_aug[np.abs(y_aug) >= threshold] * dyn_change
    )
    return np.clip(y_aug, -1, 1)


def calc(signal, seed, add_uniform=False):
    random.seed(seed)

    choice = random.randint(0, 9)
    print('choice', choice)
    if choice == 0:

        x = augmentation.sox_augment_high(
            signal,
            min_bass_gain=random.randint(25, 50),
            reverberance=random.randint(0, 80),
            hf_damping=10,
            room_scale=random.randint(0, 50),
            negate=1,
        )
    if choice == 1:
        x = augmentation.sox_augment_high(
            signal,
            min_bass_gain=random.randint(25, 70),
            reverberance=random.randint(0, 80),
            hf_damping=10,
            room_scale=random.randint(0, 50),
            negate=0,
        )
    if choice == 2:
        x = augmentation.sox_augment_low(
            signal,
            min_bass_gain=random.randint(5, 30),
            reverberance=random.randint(0, 80),
            hf_damping=10,
            room_scale=random.randint(0, 50),
            negate=random.randint(0, 1),
        )
    if choice == 3:
        x = augmentation.sox_augment_combine(
            signal,
            min_bass_gain_high=random.randint(25, 70),
            min_bass_gain_low=random.randint(5, 30),
            reverberance=random.randint(0, 80),
            hf_damping=10,
            room_scale=random.randint(0, 90),
        )
    if choice == 4:
        x = augmentation.sox_reverb(
            signal,
            reverberance=random.randint(10, 80),
            hf_damping=10,
            room_scale=random.randint(10, 90),
        )
    if choice == 5:
        x = random_amplitude_threshold(
            signal, threshold=random.uniform(0.35, 0.8)
        )
    if choice == 6:
        x = augmentation.lowpass_filter(
            signal, sr=44100, cutoff=random.randint(400, 1102)
        )
    if choice == 7:
        x = augmentation.highpass_filter(
            signal, sr=44100, cutoff=random.randint(1102, 3306)
        )
    if choice == 8:
        x = augmentation.bandpass_filter(
            signal,
            sr=44100,
            cutoff_low=random.randint(400, 1102),
            cutoff_high=random.randint(1102, 3306),
        )
    if choice == 9:
        x = signal

    if choice not in [5] and random.gauss(0.5, 0.14) > 0.6:
        x = random_amplitude_threshold(
            x, low=1.0, high=2.0, threshold=random.uniform(0.6, 0.9)
        )

    if random.gauss(0.5, 0.14) > 0.6 and add_uniform:
        x = augmentation.add_uniform_noise(
            x, power=random.uniform(0.005, 0.015)
        )

    return x


def parallel(f):
    if random.gauss(0.5, 0.14) > 0.6:
        s = random.sample(files, random.randint(2, 6))
        y = combine_speakers(s, len(s))[0]
    else:
        y = random_sampling(
            read_wav(f)[0], length=random.randint(30000, 100_000)
        )

    y = y / (np.max(np.abs(y)) + 1e-9)

    seed = random.randint(0, 100_000_000)
    x = calc(y, seed)
    if random.gauss(0.5, 0.14) > 0.6:
        print('add small noise')
        n = combine_speakers(noises, random.randint(1, 20))[0]
        n = calc(n, seed, True)
        combined, noise = augmentation.add_noise(
            x, n, factor=random.uniform(0.01, 0.1), return_noise=True
        )
    else:
        x = x / (np.max(np.abs(x)) + 1e-9)
        combined = x
    noise = combined - y
    return combined, y, noise


def loop(files):
    files = files[0]
    results = []
    for f in files:
        results.append(parallel(f))
    return results


def generate(batch_size=10, repeat=20):
    while True:
        fs = [next(file_cycle) for _ in range(batch_size)]
        results = multiprocessing(fs, loop, cores=len(fs))
        for _ in range(repeat):
            random.shuffle(results)
            for r in results:
                if (
                    not np.isnan(r[0]).any()
                    and not np.isnan(r[1]).any()
                    and not np.isnan(r[2]).any()
                ):
                    yield {'combined': r[0], 'y': r[1], 'noise': r[2]}


def get_dataset():
    def get():
        dataset = tf.data.Dataset.from_generator(
            generate,
            {'combined': tf.float32, 'y': tf.float32, 'noise': tf.float32},
            output_shapes={
                'combined': tf.TensorShape([None]),
                'y': tf.TensorShape([None]),
                'noise': tf.TensorShape([None]),
            },
        )
        return dataset

    return get


class Model:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        stft_X, D_X = tf_featurization.get_stft(self.X)

        self.stft = []
        for i in range(len(self.Y)):
            self.stft.append(tf_featurization.get_stft(self.Y[i]))

        self.outputs = []
        for i in range(len(self.Y)):
            with tf.variable_scope(f'model_{i}'):
                self.outputs.append(
                    unet.Model(D_X, dropout=0.0, training=True).logits
                )

        self.loss = []
        for i in range(len(self.Y)):
            self.loss.append(
                tf.reduce_mean(tf.abs(self.outputs[i] - self.stft[i][1]))
            )

        self.cost = tf.reduce_sum(self.loss)


init_lr = 1e-5
epochs = 500_000
init_checkpoint = 'noise-reduction-unet-output/model.ckpt'


def model_fn(features, labels, mode, params):
    model = Model(features['combined'], [features['y'], features['noise']])
    loss = model.cost

    tf.identity(loss, 'total_loss')
    tf.summary.scalar('total_loss', loss)
    for i in range(len(model.loss)):
        tf.identity(model.loss[i], f'loss_{i}')
        tf.summary.scalar(f'loss_{i}', model.loss[i])

    global_step = tf.train.get_or_create_global_step()

    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    assignment_map, initialized_variable_names = train.get_assignment_map_from_checkpoint(
        variables, init_checkpoint
    )

    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        epochs,
        end_learning_rate=1e-7,
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
        ['total_loss', 'loss_0', 'loss_1'], every_n_iter=1
    )
]
train_dataset = get_dataset()

save_directory = 'speech-enhancement-unet'

train.run_training(
    train_fn=train_dataset,
    model_fn=model_fn,
    model_dir=save_directory,
    num_gpus=2,
    log_step=1,
    save_checkpoint_step=3000,
    max_steps=epochs,
    train_hooks=train_hooks,
    eval_step=0,
)
