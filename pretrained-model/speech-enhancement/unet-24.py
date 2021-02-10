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
import malaya_speech.train.model.unet_enhancement as unet
from malaya_speech.train.model import enhancement
from malaya_speech.utils import tf_featurization
import malaya_speech.train as train
import random
from glob import glob
from itertools import cycle
from multiprocessing import Pool
import itertools


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


files = glob('/home/husein/youtube/clean-wav-22k/*.wav')
random.shuffle(files)
file_cycle = cycle(files)

noises = glob('/home/husein/youtube/noise-22k/*.wav')
random.shuffle(noises)

Y_files = glob('output-noise-reduction/*-y.wav')
Y_files = cycle(Y_files)

sr = 22050
partition_size = 4096
length = 5500


def get_pair(f):
    return f.split('/')[-1].split('-')[0]


def read_wav(f):
    return malaya_speech.load(f, sr = sr)


def random_sampling(s, length):
    return augmentation.random_sampling(s, sr = sr, length = length)


def random_amplitude(sample, low = 3, high = 5):
    y_aug = sample.copy()
    dyn_change = np.random.uniform(low = low, high = high)
    y_aug = y_aug * dyn_change
    return np.clip(y_aug, -1, 1)


def random_amplitude_threshold(sample, low = 1, high = 2, threshold = 0.4):
    y_aug = sample.copy()
    dyn_change = np.random.uniform(low = low, high = high)
    y_aug[np.abs(y_aug) >= threshold] = (
        y_aug[np.abs(y_aug) >= threshold] * dyn_change
    )
    return np.clip(y_aug, -1, 1)


def add_uniform_noise(
    sample, power = 0.01, return_noise = False, scale = False
):
    y_noise = sample.copy()
    noise_amp = power * np.random.uniform() * np.amax(y_noise)
    noise = noise_amp * np.random.normal(size = y_noise.shape[0])
    y_noise = y_noise + noise
    if scale:
        y_noise = y_noise / (np.max(np.abs(y_noise)) + 1e-9)
    if return_noise:
        if scale:
            noise = noise / (np.max(np.abs(y_noise)) + 1e-9)
        return y_noise, noise
    else:
        return y_noise


def calc(signal, seed, add_uniform = False):
    random.seed(seed)
    choice = random.randint(0, 9)
    print('choice', choice)
    if choice == 0:

        x = augmentation.sox_augment_high(
            signal,
            min_bass_gain = random.randint(25, 50),
            reverberance = random.randint(0, 80),
            hf_damping = 10,
            room_scale = random.randint(0, 50),
            negate = 1,
        )
    if choice == 1:
        x = augmentation.sox_augment_high(
            signal,
            min_bass_gain = random.randint(25, 70),
            reverberance = random.randint(0, 80),
            hf_damping = 10,
            room_scale = random.randint(0, 50),
            negate = 0,
        )
    if choice == 2:
        x = augmentation.sox_augment_low(
            signal,
            min_bass_gain = random.randint(5, 30),
            reverberance = random.randint(0, 80),
            hf_damping = 10,
            room_scale = random.randint(0, 50),
            negate = random.randint(0, 1),
        )
    if choice == 3:
        x = augmentation.sox_augment_combine(
            signal,
            min_bass_gain_high = random.randint(25, 70),
            min_bass_gain_low = random.randint(5, 30),
            reverberance = random.randint(0, 80),
            hf_damping = 10,
            room_scale = random.randint(0, 90),
        )
    if choice == 4:
        x = augmentation.sox_reverb(
            signal,
            reverberance = random.randint(10, 80),
            hf_damping = 10,
            room_scale = random.randint(10, 90),
        )
    if choice == 5:
        x = random_amplitude_threshold(
            signal, threshold = random.uniform(0.35, 0.8)
        )
    if choice == 6:
        x = augmentation.lowpass_filter(
            signal, sr = sr, cutoff = random.randint(200, 551)
        )
    if choice == 7:
        x = augmentation.highpass_filter(
            signal, sr = sr, cutoff = random.randint(551, 1653)
        )
    if choice == 8:
        x = augmentation.bandpass_filter(
            signal,
            sr = sr,
            cutoff_low = random.randint(200, 551),
            cutoff_high = random.randint(551, 1653),
        )
    if choice == 9:
        x = signal

    if choice not in [5] and random.gauss(0.5, 0.14) > 0.6:
        x = random_amplitude_threshold(
            x, low = 1.0, high = 2.0, threshold = random.uniform(0.6, 0.9)
        )

    if random.gauss(0.5, 0.14) > 0.6 and add_uniform:
        x = add_uniform_noise(x, power = random.uniform(0.005, 0.015))

    return x


def combine_speakers(files, n = 5):
    w_samples = random.sample(files, n)
    w_samples = [
        random_sampling(read_wav(f)[0], length = length // 10)
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


def parallel(f):
    y = random_sampling(read_wav(f)[0], length = length)
    seed = random.randint(0, 100_000_000)
    x = calc(y, seed)
    if random.gauss(0.5, 0.14) > 0.6:
        print('add small noise')
        n = combine_speakers(noises, random.randint(1, 20))[0]
        n = calc(n, seed, True)
        combined, noise = augmentation.add_noise(
            x,
            n,
            factor = random.uniform(0.01, 0.1),
            return_noise = True,
            rescale = False,
        )
    else:
        combined = x
    noise = combined - y
    return combined, y, noise


def parallel_semisupervised(f):
    f_ = get_pair(f)
    f_ = f'output-noise-reduction/{f_}-y_.wav'
    y = read_wav(f)[0]
    combined = read_wav(f_)[0]
    sr_ = int(sr / 1000)
    up = len(y) - (sr_ * length)
    if up < 1:
        r = 0
    else:
        r = np.random.randint(0, up)
    y = y[r : r + sr_ * length]
    combined = combined[r : r + sr_ * length]
    noise = combined - y
    return combined, y, noise


def loop(files):
    files = files[0]
    results = []
    for no, f in enumerate(files):
        results.append(parallel(f))
    return results


def loop_semisupervised(files):
    files = files[0]
    results = []
    for f in files:
        results.append(parallel_semisupervised(f))
    return results


def generate(batch_size = 10, repeat = 30):
    while True:
        fs = [next(file_cycle) for _ in range(batch_size)]
        results = multiprocessing(fs, loop, cores = len(fs))
        fs = [next(Y_files) for _ in range(batch_size)]
        results.extend(
            multiprocessing(fs, loop_semisupervised, cores = len(fs))
        )
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
            output_shapes = {
                'combined': tf.TensorShape([None]),
                'y': tf.TensorShape([None]),
                'noise': tf.TensorShape([None]),
            },
        )
        return dataset

    return get


init_lr = 1e-4
epochs = 2_000_000


def model_fn(features, labels, mode, params):

    x = tf.expand_dims(features['combined'], -1)
    y = tf.expand_dims(features['y'], -1)
    partitioned_x = tf_featurization.pad_and_partition(x, partition_size)
    partitioned_y = tf_featurization.pad_and_partition(y, partition_size)
    model = unet.Model(partitioned_x, channels_interval = 24)
    l2_loss, snr = enhancement.loss.snr(model.logits, partitioned_y)
    sdr = enhancement.loss.sdr(model.logits, partitioned_y)
    mae = tf.losses.absolute_difference
    mae_loss = mae(labels = partitioned_y, predictions = model.logits)
    loss = mae_loss

    tf.identity(loss, 'total_loss')
    tf.summary.scalar('total_loss', loss)

    tf.summary.scalar('snr', snr)
    tf.summary.scalar('sdr', sdr)

    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.constant(value = init_lr, shape = [], dtype = tf.float32)
    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        epochs,
        end_learning_rate = 1e-6,
        power = 1.0,
        cycle = False,
    )
    tf.summary.scalar('learning_rate', learning_rate)

    if mode == tf.estimator.ModeKeys.TRAIN:

        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

        train_op = optimizer.minimize(loss, global_step = global_step)
        estimator_spec = tf.estimator.EstimatorSpec(
            mode = mode, loss = loss, train_op = train_op
        )

    elif mode == tf.estimator.ModeKeys.EVAL:

        estimator_spec = tf.estimator.EstimatorSpec(
            mode = tf.estimator.ModeKeys.EVAL, loss = loss
        )

    return estimator_spec


train_hooks = [tf.train.LoggingTensorHook(['total_loss'], every_n_iter = 1)]
train_dataset = get_dataset()

save_directory = 'speech-enhancement-unet-24'

train.run_training(
    train_fn = train_dataset,
    model_fn = model_fn,
    model_dir = save_directory,
    num_gpus = 1,
    log_step = 1,
    save_checkpoint_step = 10000,
    max_steps = epochs,
    train_hooks = train_hooks,
    eval_step = 0,
)
