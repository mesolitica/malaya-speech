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
import pyroomacoustics as pra
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

sr = 22050
partition_size = 4096


def read_wav(f):
    return malaya_speech.load(f, sr = sr)


def random_sampling(s, length):
    return augmentation.random_sampling(s, sr = sr, length = length)


def augment_room(y, scale = 1.0):
    corners = np.array(
        [[0, 0], [0, 5 * scale], [3 * scale, 5 * scale], [3 * scale, 0]]
    ).T
    room = pra.Room.from_corners(
        corners,
        fs = sr,
        materials = pra.Material(0.2, 0.15),
        ray_tracing = True,
        air_absorption = True,
    )
    room.extrude(2.0, materials = pra.Material(0.2, 0.15))
    room.set_ray_tracing(
        receiver_radius = 0.5, n_rays = 10000, energy_thres = 1e-5
    )
    room.add_source([1.5 * scale, 4 * scale, 0.5], signal = y)
    R = np.array([[1.5 * scale], [0.5 * scale], [0.5]])
    room.add_microphone(R)
    room.simulate()
    return room.mic_array.signals[0]


def combine_speakers(files, n = 5):
    w_samples = random.sample(files, n)

    w_samples = [
        random_sampling(read_wav(f)[0], length = random.randint(1200, 5000))
        for f in w_samples
    ]
    left = w_samples[0].copy()

    if random.gauss(0.5, 0.14) > 0.5:
        s = random.uniform(0.5, 3.0)
        a = random.uniform(0.5, 1.0)
        print(0, s, a)
        left = augment_room(left, scale = s) * a

    left_actual = w_samples[0].copy()
    left = left[: len(left_actual)]

    for i in range(1, n):
        right = w_samples[i].copy()

        if random.gauss(0.5, 0.14) > 0.5:
            s = random.uniform(0.5, 3.0)
            a = random.uniform(0.5, 1.0)
            right_y = augment_room(right, scale = s)
            right_y = right_y * a
            print(i, s, a)
        else:
            right_y = right

        print(len(right_y), len(right))

        right_y = right_y[: len(right)]

        print(len(right_y), len(right))

        overlap = random.uniform(0.5, 1.25)
        left_len = int(overlap * len(left))

        padded_right = np.pad(right, (left_len, 0))
        padded_right_y = np.pad(right_y, (left_len, 0))

        if len(left) > len(padded_right):
            padded_right = np.pad(
                padded_right, (0, len(left_actual) - len(padded_right))
            )
            padded_right_y = np.pad(
                padded_right_y, (0, len(left) - len(padded_right_y))
            )
        else:
            left = np.pad(left, (0, len(padded_right_y) - len(left)))
            left_actual = np.pad(
                left_actual, (0, len(padded_right) - len(left_actual))
            )

        left_actual = left_actual + padded_right
        left = left + padded_right_y
    return left, left_actual


def parallel(f):
    combined, y = combine_speakers(files, random.randint(1, 6))
    noise = combined - y
    combined = combined / (np.max(np.abs(combined)) + 1e-9)
    y = y / (np.max(np.abs(y)) + 1e-9)
    return combined, y, noise


def loop(files):
    files = files[0]
    results = []
    for no, f in enumerate(files):
        results.append(parallel(f))
    return results


def generate(batch_size = 10, repeat = 30):
    while True:
        fs = [next(file_cycle) for _ in range(batch_size)]
        results = multiprocessing(fs, loop, cores = len(fs))
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

save_directory = 'background-enhance-unet-24'

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
