import os
import warnings

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings('ignore')

import tensorflow as tf
import malaya_speech.augmentation.waveform as augmentation
from malaya_speech.train.model import super_res
from malaya_speech.train.model import enhancement
import malaya_speech
import malaya_speech.train as train
from glob import glob
import random
import numpy as np
import IPython.display as ipd
from multiprocessing import Pool
from itertools import cycle
import itertools

np.seterr(all='raise')


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
noises = [n for n in noises if (os.path.getsize(n) / 1e6) < 50]
file_cycle = cycle(files)
sr = 44100
selected_sr = [6000, 8000, 16000]


def read_wav(f):
    return malaya_speech.load(f, sr=sr)


def random_sampling(s, length):
    return augmentation.random_sampling(s, sr=sr, length=length)


def downsample(y, sr, down_sr):
    y_ = malaya_speech.resample(y, sr, down_sr)
    return malaya_speech.resample(y_, down_sr, sr)


def parallel(f):
    y = read_wav(f)[0]
    y = random_sampling(y, length=random.randint(3000, 10000))
    y = y / (np.max(np.abs(y)) + 1e-9)
    y_ = downsample(y, sr, random.choice(selected_sr))
    return y_, y


def loop(files):
    files = files[0]
    results = []
    for f in files:
        results.append(parallel(f))
    return results


def generate(batch_size=10, repeat=5):
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


init_lr = 2e-4
epochs = 500_000
partition = 8192


def model_fn(features, labels, mode, params):

    combined = tf.expand_dims(features['combined'], -1)
    y = tf.expand_dims(features['y'], -1)
    partitioned_x = malaya_speech.tf_featurization.pad_and_partition(
        combined, partition
    )
    partitioned_y = malaya_speech.tf_featurization.pad_and_partition(
        y, partition
    )
    model = super_res.UNET(partitioned_x, dropout=0.0)
    l2_loss, snr = enhancement.loss.snr(model.logits, partitioned_y)
    sdr = enhancement.loss.sdr(model.logits, partitioned_y)
    loss = l2_loss

    tf.identity(loss, 'total_loss')

    tf.summary.scalar('total_loss', loss)
    tf.summary.scalar('snr', snr)
    tf.summary.scalar('sdr', sdr)

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

        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate, beta1=0.99, beta2=0.999
        )

        train_op = optimizer.minimize(loss, global_step=global_step)
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

save_directory = 'super-resolution-unet'

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
