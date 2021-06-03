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
import malaya_speech.train as train
import random
from glob import glob
from itertools import cycle
from multiprocessing import Pool
import itertools
from malaya_speech.utils import tf_featurization
from tensorflow.signal import stft, inverse_stft, hann_window


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
sr = 44100


def read_wav(f):
    return malaya_speech.load(f, sr=sr)


def random_sampling(s, length):
    return augmentation.random_sampling(s, sr=sr, length=length)


def random_cut(sample, length=500):
    up = len(sample) - length
    if up < 1:
        r = 0
    else:
        r = np.random.randint(0, up)
    return sample[r: r + length]


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


def downsample(y, sr, down_sr):
    y_ = malaya_speech.resample(y, sr, down_sr)
    return malaya_speech.resample(y_, down_sr, sr)


def calc(signal, seed, add_uniform=False):
    random.seed(seed)

    if not add_uniform:
        signal = downsample(signal, sr, random.randint(8000, 14000))

    choice = random.randint(0, 10)
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

    if choice > 5:
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
        y = read_wav(f)[0]

    y = random_sampling(y, length=random.randint(2000, 300_000))
    y = y / (np.max(np.abs(y)) + 1e-9)

    seed = random.randint(0, 100_000_000)
    x = calc(y.copy(), seed)
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
    print(combined.shape, y.shape)
    return combined, y


def loop(files):
    files = files[0]
    results = []
    for f in files:
        results.append(parallel(f))
    return results


def generate(batch_size=10, repeat=30):
    while True:
        fs = [next(file_cycle) for _ in range(batch_size)]
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
                'y': tf.TensorShape([None]),
            },
        )
        return dataset

    return get


init_lr = 5e-4
epochs = 500_000


class Model:
    def __init__(self, X, Y, frame_length=4096, frame_step=1024):
        def get_stft(X):
            return tf.signal.stft(
                X,
                frame_length,
                frame_step,
                window_fn=lambda frame_length, dtype: (
                    hann_window(frame_length, periodic=True, dtype=dtype)
                ),
                pad_end=True,
            )

        stft_X = get_stft(X)
        stft_Y = get_stft(Y)
        mag_X = tf.abs(stft_X)
        mag_Y = tf.abs(stft_Y)

        angle_X = tf.math.imag(stft_X)
        angle_Y = tf.math.imag(stft_Y)

        partitioned_mag_X = tf_featurization.pad_and_partition(mag_X, 512)
        partitioned_angle_X = tf_featurization.pad_and_partition(angle_X, 512)
        params = {'conv_n_filters': [32 * (2 ** i) for i in range(6)]}

        with tf.variable_scope('model_mag'):
            mix_mag = tf.expand_dims(partitioned_mag_X, 3)[:, :, :-1, :]
            mix_mag_logits = unet.Model(
                mix_mag,
                output_mask_logit=True,
                dropout=0.0,
                training=True,
                params=params,
            ).logits
            mix_mag_logits = tf.squeeze(mix_mag_logits, 3)
            mix_mag_logits = tf.pad(
                mix_mag_logits, [(0, 0), (0, 0), (0, 1)], mode='CONSTANT'
            )
            mix_mag_logits = tf.nn.relu(mix_mag_logits)

        with tf.variable_scope('model_angle'):
            mix_angle = tf.expand_dims(partitioned_angle_X, 3)[:, :, :-1, :]
            mix_angle_logits = unet.Model(
                mix_angle,
                output_mask_logit=True,
                dropout=0.0,
                training=True,
                params=params,
            ).logits
            mix_angle_logits = tf.squeeze(mix_angle_logits, 3)
            mix_angle_logits = tf.pad(
                mix_angle_logits, [(0, 0), (0, 0), (0, 1)], mode='CONSTANT'
            )

        partitioned_mag_Y = tf_featurization.pad_and_partition(mag_Y, 512)
        partitioned_angle_Y = tf_featurization.pad_and_partition(angle_Y, 512)

        self.mag_l1 = tf.reduce_mean(tf.abs(partitioned_mag_Y - mix_mag_logits))
        self.angle_l1 = tf.reduce_mean(
            tf.abs(partitioned_angle_Y - mix_angle_logits)
        )
        self.cost = self.mag_l1 + self.angle_l1

        def get_original_shape(D, stft):
            instrument_mask = D

            old_shape = tf.shape(instrument_mask)
            new_shape = tf.concat(
                [[old_shape[0] * old_shape[1]], old_shape[2:]], axis=0
            )
            instrument_mask = tf.reshape(instrument_mask, new_shape)
            instrument_mask = instrument_mask[: tf.shape(stft)[0]]
            return instrument_mask

        _mag = get_original_shape(tf.expand_dims(mix_mag_logits, -1), stft_X)
        _angle = get_original_shape(
            tf.expand_dims(mix_angle_logits, -1), stft_X
        )

        stft = tf.multiply(
            tf.complex(_mag, 0.0), tf.exp(tf.complex(0.0, _angle))
        )

        inverse_stft_X = inverse_stft(
            stft[:, :, 0],
            frame_length,
            frame_step,
            window_fn=lambda frame_length, dtype: (
                hann_window(frame_length, periodic=True, dtype=dtype)
            ),
        )


def model_fn(features, labels, mode, params):

    model = Model(features['combined'], features['y'])

    mag_l1 = model.mag_l1
    angle_l1 = model.angle_l1
    loss = model.cost

    tf.identity(loss, 'total_loss')
    tf.identity(mag_l1, 'mag_l1')
    tf.identity(angle_l1, 'angle_l1')

    tf.summary.scalar('total_loss', loss)
    tf.summary.scalar('mag_l1', mag_l1)
    tf.summary.scalar('angle_l1', angle_l1)

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
        ['total_loss', 'mag_l1', 'angle_l1'], every_n_iter=1
    )
]
train_dataset = get_dataset()

save_directory = 'speech-enhancement-unet-spectrogram'

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
