import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf

import collections
import re
import random


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match('^(.*):\\d+$', name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue

        assignment_map[name] = name
        assignment_map[name] = name_to_variable[name]
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ':0'] = 1

    tf.logging.info('**** Trainable Variables ****')
    for var in tvars:
        init_string = ''
        if var.name in initialized_variable_names:
            init_string = ', *INIT_FROM_CKPT*'
        tf.logging.info(
            '  name = %s, shape = %s%s', var.name, var.shape, init_string
        )

    return (assignment_map, initialized_variable_names)


import malaya_speech.train as train
import malaya_speech
from glob import glob

import librosa
import numpy as np
from scipy.signal import lfilter, butter
import decimal
import math

dimension = 512

# for VGGVox v1
def round_half_up(number):
    return int(
        decimal.Decimal(number).quantize(
            decimal.Decimal('1'), rounding = decimal.ROUND_HALF_UP
        )
    )


# for VGGVox v1
def remove_dc_and_dither(sin, sample_rate):
    if sample_rate == 16e3:
        alpha = 0.99
    elif sample_rate == 8e3:
        alpha = 0.999
    else:
        print('Sample rate must be 16kHz or 8kHz only')
        exit(1)
    sin = lfilter([1, -1], [1, -alpha], sin)
    dither = (
        np.random.random_sample(len(sin))
        + np.random.random_sample(len(sin))
        - 1
    )
    spow = np.std(dither)
    sout = sin + 1e-6 * spow * dither
    return sout


# for VGGVox v1
def preemphasis(signal, coeff = 0.95):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


# for VGGVox v1
def rolling_window(a, window, step = 1):
    # http://ellisvalentiner.com/post/2017-03-21-np-strides-trick
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape = shape, strides = strides)[
        ::step
    ]


# for VGGVox v1
def framesig(
    sig,
    frame_len,
    frame_step,
    winfunc = lambda x: numpy.ones((x,)),
    stride_trick = True,
):
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(
            math.ceil((1.0 * slen - frame_len) / frame_step)
        )  # LV

    padlen = int((numframes - 1) * frame_step + frame_len)

    zeros = np.zeros((padlen - slen,))
    padsignal = np.concatenate((sig, zeros))
    if stride_trick:
        win = winfunc(frame_len)
        frames = rolling_window(
            padsignal, window = frame_len, step = frame_step
        )
    else:
        indices = (
            numpy.tile(numpy.arange(0, frame_len), (numframes, 1))
            + numpy.tile(
                numpy.arange(0, numframes * frame_step, frame_step),
                (frame_len, 1),
            ).T
        )
        indices = numpy.array(indices, dtype = numpy.int32)
        frames = padsignal[indices]
        win = numpy.tile(winfunc(frame_len), (numframes, 1))

    return frames * win


# for VGGVox v1
def normalize_frames(m, epsilon = 1e-12):
    return np.array([(v - np.mean(v)) / max(np.std(v), epsilon) for v in m])


# for VGGVox v1
def vggvox_v1(
    signal,
    sample_rate = 16000,
    preemphasis_alpha = 0.97,
    frame_len = 0.005,
    frame_step = 0.0005,
    num_fft = 512,
    buckets = None,
    **kwargs
):
    signal = signal.copy()
    signal *= 2 ** 15
    signal = remove_dc_and_dither(signal, sample_rate)
    signal = preemphasis(signal, coeff = preemphasis_alpha)
    frames = framesig(
        signal,
        frame_len = frame_len * sample_rate,
        frame_step = frame_step * sample_rate,
        winfunc = np.hamming,
    )
    fft = abs(np.fft.fft(frames, n = num_fft))
    fft_norm = normalize_frames(fft.T)

    if buckets:
        rsize = max(k for k in buckets if k <= fft_norm.shape[1])
        rstart = int((fft_norm.shape[1] - rsize) / 2)
        out = fft_norm[:, rstart : rstart + rsize]
        return out

    else:
        return fft_norm.astype('float32')


def calc(v):
    r = vggvox_v1(v)
    return r


def preprocess_inputs(example):
    s = tf.compat.v1.numpy_function(calc, [example['waveforms']], tf.float32)

    s = tf.reshape(s, (dimension, -1, 1))
    example['inputs'] = s

    return example


def parse(serialized_example):

    data_fields = {
        'waveforms': tf.VarLenFeature(tf.float32),
        'targets': tf.VarLenFeature(tf.int64),
    }
    features = tf.parse_single_example(
        serialized_example, features = data_fields
    )
    for k in features.keys():
        features[k] = features[k].values

    features = preprocess_inputs(features)

    keys = list(features.keys())
    for k in keys:
        if k not in ['inputs', 'targets']:
            features.pop(k, None)

    return features


def get_dataset(files, batch_size = 16, shuffle_size = 5, thread_count = 24):
    def get():
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.map(parse, num_parallel_calls = thread_count)
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes = {
                'inputs': tf.TensorShape([dimension, None, 1]),
                'targets': tf.TensorShape([None]),
            },
            padding_values = {
                'inputs': tf.constant(0, dtype = tf.float32),
                'targets': tf.constant(0, dtype = tf.int64),
            },
        )
        dataset = dataset.shuffle(shuffle_size)
        dataset = dataset.repeat()
        return dataset

    return get


import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Reshape
from tensorflow.keras.layers import (
    Conv2D,
    ZeroPadding2D,
    MaxPooling2D,
    AveragePooling2D,
)
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Lambda, Activation
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


def conv_bn_pool(
    inp_tensor,
    layer_idx,
    conv_filters,
    conv_kernel_size,
    conv_strides,
    conv_pad,
    pool = '',
    pool_size = (2, 2),
    pool_strides = None,
    conv_layer_prefix = 'conv',
):
    x = ZeroPadding2D(padding = conv_pad, name = 'pad{}'.format(layer_idx))(
        inp_tensor
    )
    x = Conv2D(
        filters = conv_filters,
        kernel_size = conv_kernel_size,
        strides = conv_strides,
        padding = 'valid',
        name = '{}{}'.format(conv_layer_prefix, layer_idx),
    )(x)
    x = BatchNormalization(
        epsilon = 1e-5, momentum = 1.0, name = 'bn{}'.format(layer_idx)
    )(x)
    x = Activation('relu', name = 'relu{}'.format(layer_idx))(x)
    if pool == 'max':
        x = MaxPooling2D(
            pool_size = pool_size,
            strides = pool_strides,
            name = 'mpool{}'.format(layer_idx),
        )(x)
    elif pool == 'avg':
        x = AveragePooling2D(
            pool_size = pool_size,
            strides = pool_strides,
            name = 'apool{}'.format(layer_idx),
        )(x)
    return x


# Block of layers: Conv --> BatchNorm --> ReLU --> Dynamic average pool (fc6 -> apool6 only)
def conv_bn_dynamic_apool(
    inp_tensor,
    layer_idx,
    conv_filters,
    conv_kernel_size,
    conv_strides,
    conv_pad,
    conv_layer_prefix = 'conv',
):
    x = ZeroPadding2D(padding = conv_pad, name = 'pad{}'.format(layer_idx))(
        inp_tensor
    )
    x = Conv2D(
        filters = conv_filters,
        kernel_size = conv_kernel_size,
        strides = conv_strides,
        padding = 'valid',
        name = '{}{}'.format(conv_layer_prefix, layer_idx),
    )(x)
    x = BatchNormalization(
        epsilon = 1e-5, momentum = 1.0, name = 'bn{}'.format(layer_idx)
    )(x)
    x = Activation('relu', name = 'relu{}'.format(layer_idx))(x)
    x = GlobalAveragePooling2D(name = 'gapool{}'.format(layer_idx))(x)
    x = Reshape((1, 1, conv_filters), name = 'reshape{}'.format(layer_idx))(x)
    return x


class Resnet1D(Model):
    def __init__(self, params = None, is_training = False):
        super(Resnet1D, self).__init__()

    def call(self, inputs, training = None, mask = None):
        inp = inputs['features_input']
        x = conv_bn_pool(
            inp,
            layer_idx = 1,
            conv_filters = 96,
            conv_kernel_size = (7, 7),
            conv_strides = (2, 2),
            conv_pad = (1, 1),
            pool = 'max',
            pool_size = (3, 3),
            pool_strides = (2, 2),
        )
        x = conv_bn_pool(
            x,
            layer_idx = 2,
            conv_filters = 256,
            conv_kernel_size = (5, 5),
            conv_strides = (2, 2),
            conv_pad = (1, 1),
            pool = 'max',
            pool_size = (3, 3),
            pool_strides = (2, 2),
        )
        x = conv_bn_pool(
            x,
            layer_idx = 3,
            conv_filters = 384,
            conv_kernel_size = (3, 3),
            conv_strides = (1, 1),
            conv_pad = (1, 1),
        )
        x = conv_bn_pool(
            x,
            layer_idx = 4,
            conv_filters = 256,
            conv_kernel_size = (3, 3),
            conv_strides = (1, 1),
            conv_pad = (1, 1),
        )
        x = conv_bn_pool(
            x,
            layer_idx = 5,
            conv_filters = 256,
            conv_kernel_size = (3, 3),
            conv_strides = (1, 1),
            conv_pad = (1, 1),
            pool = 'max',
            pool_size = (5, 3),
            pool_strides = (3, 2),
        )
        x = conv_bn_dynamic_apool(
            x,
            layer_idx = 6,
            conv_filters = 4096,
            conv_kernel_size = (9, 1),
            conv_strides = (1, 1),
            conv_pad = (0, 0),
            conv_layer_prefix = 'fc',
        )
        x = conv_bn_pool(
            x,
            layer_idx = 7,
            conv_filters = 1024,
            conv_kernel_size = (1, 1),
            conv_strides = (1, 1),
            conv_pad = (0, 0),
            conv_layer_prefix = 'fc',
        )
        x = Lambda(lambda y: K.l2_normalize(y, axis = 3), name = 'norm')(x)
        x = Conv2D(
            filters = 1024,
            kernel_size = (1, 1),
            strides = (1, 1),
            padding = 'valid',
            name = 'fc8',
        )(x)
        return x


learning_rate = 1e-5
init_checkpoint = '../vggvox-speaker-identification/v1/vggvox.ckpt'


def model_fn(features, labels, mode, params):
    Y = tf.cast(features['targets'][:, 0], tf.int32)
    model = Resnet1D(is_training = True)
    inputs = {'features_input': features['inputs']}

    logits = model.call(inputs)
    logits = logits[:, 0, 0, :]
    logits = tf.layers.dense(logits, 2)

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

    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    variables = [v for v in variables if 'prediction' not in v.name]

    assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(
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

train_files = glob('vad2/data/vad-train-*') + glob('noise/data/vad-train-*')
random.shuffle(train_files)
train_dataset = get_dataset(train_files, batch_size = 32)

dev_files = glob('vad2/data/vad-dev-*') + glob('noise/data/vad-dev-*')
random.shuffle(dev_files)
dev_dataset = get_dataset(dev_files, batch_size = 16)

save_directory = 'output-vggvox-v1-vad'

train.run_training(
    train_fn = train_dataset,
    model_fn = model_fn,
    model_dir = save_directory,
    num_gpus = 1,
    log_step = 1,
    save_checkpoint_step = 10000,
    max_steps = 300000,
    eval_fn = dev_dataset,
    train_hooks = train_hooks,
)
