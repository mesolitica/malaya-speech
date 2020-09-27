import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

import tensorflow as tf
import numpy as np

# !wget https://raw.githubusercontent.com/tensorflow/models/master/research/slim/nets/inception_utils.py

import tensorflow.compat.v1 as tf
import tf_slim as slim
import inception_utils


def block_inception_a(inputs, scope = None, reuse = None):
    """Builds Inception-A block for Inception v4 network."""
    # By default use stride=1 and SAME padding
    with slim.arg_scope(
        [slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
        stride = 1,
        padding = 'SAME',
    ):
        with tf.variable_scope(
            scope, 'BlockInceptionA', [inputs], reuse = reuse
        ):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(
                    inputs, 96, [1, 1], scope = 'Conv2d_0a_1x1'
                )
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(
                    inputs, 64, [1, 1], scope = 'Conv2d_0a_1x1'
                )
                branch_1 = slim.conv2d(
                    branch_1, 96, [3, 3], scope = 'Conv2d_0b_3x3'
                )
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(
                    inputs, 64, [1, 1], scope = 'Conv2d_0a_1x1'
                )
                branch_2 = slim.conv2d(
                    branch_2, 96, [3, 3], scope = 'Conv2d_0b_3x3'
                )
                branch_2 = slim.conv2d(
                    branch_2, 96, [3, 3], scope = 'Conv2d_0c_3x3'
                )
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(
                    inputs, [3, 3], scope = 'AvgPool_0a_3x3'
                )
                branch_3 = slim.conv2d(
                    branch_3, 96, [1, 1], scope = 'Conv2d_0b_1x1'
                )
            return tf.concat(
                axis = 3, values = [branch_0, branch_1, branch_2, branch_3]
            )


def block_reduction_a(inputs, scope = None, reuse = None):
    """Builds Reduction-A block for Inception v4 network."""
    # By default use stride=1 and SAME padding
    with slim.arg_scope(
        [slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
        stride = 1,
        padding = 'SAME',
    ):
        with tf.variable_scope(
            scope, 'BlockReductionA', [inputs], reuse = reuse
        ):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(
                    inputs,
                    384,
                    [3, 3],
                    stride = 2,
                    padding = 'VALID',
                    scope = 'Conv2d_1a_3x3',
                )
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(
                    inputs, 192, [1, 1], scope = 'Conv2d_0a_1x1'
                )
                branch_1 = slim.conv2d(
                    branch_1, 224, [3, 3], scope = 'Conv2d_0b_3x3'
                )
                branch_1 = slim.conv2d(
                    branch_1,
                    256,
                    [3, 3],
                    stride = 2,
                    padding = 'VALID',
                    scope = 'Conv2d_1a_3x3',
                )
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.max_pool2d(
                    inputs,
                    [3, 3],
                    stride = 2,
                    padding = 'VALID',
                    scope = 'MaxPool_1a_3x3',
                )
            return tf.concat(axis = 3, values = [branch_0, branch_1, branch_2])


def block_inception_b(inputs, scope = None, reuse = None):
    """Builds Inception-B block for Inception v4 network."""
    # By default use stride=1 and SAME padding
    with slim.arg_scope(
        [slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
        stride = 1,
        padding = 'SAME',
    ):
        with tf.variable_scope(
            scope, 'BlockInceptionB', [inputs], reuse = reuse
        ):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(
                    inputs, 384, [1, 1], scope = 'Conv2d_0a_1x1'
                )
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(
                    inputs, 192, [1, 1], scope = 'Conv2d_0a_1x1'
                )
                branch_1 = slim.conv2d(
                    branch_1, 224, [1, 7], scope = 'Conv2d_0b_1x7'
                )
                branch_1 = slim.conv2d(
                    branch_1, 256, [7, 1], scope = 'Conv2d_0c_7x1'
                )
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(
                    inputs, 192, [1, 1], scope = 'Conv2d_0a_1x1'
                )
                branch_2 = slim.conv2d(
                    branch_2, 192, [7, 1], scope = 'Conv2d_0b_7x1'
                )
                branch_2 = slim.conv2d(
                    branch_2, 224, [1, 7], scope = 'Conv2d_0c_1x7'
                )
                branch_2 = slim.conv2d(
                    branch_2, 224, [7, 1], scope = 'Conv2d_0d_7x1'
                )
                branch_2 = slim.conv2d(
                    branch_2, 256, [1, 7], scope = 'Conv2d_0e_1x7'
                )
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(
                    inputs, [3, 3], scope = 'AvgPool_0a_3x3'
                )
                branch_3 = slim.conv2d(
                    branch_3, 128, [1, 1], scope = 'Conv2d_0b_1x1'
                )
            return tf.concat(
                axis = 3, values = [branch_0, branch_1, branch_2, branch_3]
            )


def block_reduction_b(inputs, scope = None, reuse = None):
    """Builds Reduction-B block for Inception v4 network."""
    # By default use stride=1 and SAME padding
    with slim.arg_scope(
        [slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
        stride = 1,
        padding = 'SAME',
    ):
        with tf.variable_scope(
            scope, 'BlockReductionB', [inputs], reuse = reuse
        ):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(
                    inputs, 192, [1, 1], scope = 'Conv2d_0a_1x1'
                )
                branch_0 = slim.conv2d(
                    branch_0,
                    192,
                    [3, 3],
                    stride = 2,
                    padding = 'VALID',
                    scope = 'Conv2d_1a_3x3',
                )
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(
                    inputs, 256, [1, 1], scope = 'Conv2d_0a_1x1'
                )
                branch_1 = slim.conv2d(
                    branch_1, 256, [1, 7], scope = 'Conv2d_0b_1x7'
                )
                branch_1 = slim.conv2d(
                    branch_1, 320, [7, 1], scope = 'Conv2d_0c_7x1'
                )
                branch_1 = slim.conv2d(
                    branch_1,
                    320,
                    [3, 3],
                    stride = 2,
                    padding = 'VALID',
                    scope = 'Conv2d_1a_3x3',
                )
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.max_pool2d(
                    inputs,
                    [3, 3],
                    stride = 2,
                    padding = 'VALID',
                    scope = 'MaxPool_1a_3x3',
                )
            return tf.concat(axis = 3, values = [branch_0, branch_1, branch_2])


def block_inception_c(inputs, scope = None, reuse = None):
    """Builds Inception-C block for Inception v4 network."""
    # By default use stride=1 and SAME padding
    with slim.arg_scope(
        [slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
        stride = 1,
        padding = 'SAME',
    ):
        with tf.variable_scope(
            scope, 'BlockInceptionC', [inputs], reuse = reuse
        ):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(
                    inputs, 256, [1, 1], scope = 'Conv2d_0a_1x1'
                )
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(
                    inputs, 384, [1, 1], scope = 'Conv2d_0a_1x1'
                )
                branch_1 = tf.concat(
                    axis = 3,
                    values = [
                        slim.conv2d(
                            branch_1, 256, [1, 3], scope = 'Conv2d_0b_1x3'
                        ),
                        slim.conv2d(
                            branch_1, 256, [3, 1], scope = 'Conv2d_0c_3x1'
                        ),
                    ],
                )
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(
                    inputs, 384, [1, 1], scope = 'Conv2d_0a_1x1'
                )
                branch_2 = slim.conv2d(
                    branch_2, 448, [3, 1], scope = 'Conv2d_0b_3x1'
                )
                branch_2 = slim.conv2d(
                    branch_2, 512, [1, 3], scope = 'Conv2d_0c_1x3'
                )
                branch_2 = tf.concat(
                    axis = 3,
                    values = [
                        slim.conv2d(
                            branch_2, 256, [1, 3], scope = 'Conv2d_0d_1x3'
                        ),
                        slim.conv2d(
                            branch_2, 256, [3, 1], scope = 'Conv2d_0e_3x1'
                        ),
                    ],
                )
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(
                    inputs, [3, 3], scope = 'AvgPool_0a_3x3'
                )
                branch_3 = slim.conv2d(
                    branch_3, 256, [1, 1], scope = 'Conv2d_0b_1x1'
                )
            return tf.concat(
                axis = 3, values = [branch_0, branch_1, branch_2, branch_3]
            )


def inception_v4_base(inputs, final_endpoint = 'Mixed_7d', scope = None):
    """Creates the Inception V4 network up to the given final endpoint.
  Args:
    inputs: a 4-D tensor of size [batch_size, height, width, 3].
    final_endpoint: specifies the endpoint to construct the network up to.
      It can be one of [ 'Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
      'Mixed_3a', 'Mixed_4a', 'Mixed_5a', 'Mixed_5b', 'Mixed_5c', 'Mixed_5d',
      'Mixed_5e', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d', 'Mixed_6e',
      'Mixed_6f', 'Mixed_6g', 'Mixed_6h', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c',
      'Mixed_7d']
    scope: Optional variable_scope.
  Returns:
    logits: the logits outputs of the model.
    end_points: the set of end_points from the inception model.
  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
  """
    end_points = {}

    def add_and_check_final(name, net):
        end_points[name] = net
        return name == final_endpoint

    with tf.variable_scope(scope, 'InceptionV4', [inputs]):
        with slim.arg_scope(
            [slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
            stride = 1,
            padding = 'SAME',
        ):
            # 299 x 299 x 3
            net = slim.conv2d(
                inputs,
                32,
                [3, 3],
                stride = 2,
                padding = 'VALID',
                scope = 'Conv2d_1a_3x3',
            )
            if add_and_check_final('Conv2d_1a_3x3', net):
                return net, end_points
            # 149 x 149 x 32
            net = slim.conv2d(
                net, 32, [3, 3], padding = 'VALID', scope = 'Conv2d_2a_3x3'
            )
            if add_and_check_final('Conv2d_2a_3x3', net):
                return net, end_points
            # 147 x 147 x 32
            net = slim.conv2d(net, 64, [3, 3], scope = 'Conv2d_2b_3x3')
            if add_and_check_final('Conv2d_2b_3x3', net):
                return net, end_points
            # 147 x 147 x 64
            with tf.variable_scope('Mixed_3a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.max_pool2d(
                        net,
                        [3, 3],
                        stride = 2,
                        padding = 'VALID',
                        scope = 'MaxPool_0a_3x3',
                    )
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(
                        net,
                        96,
                        [3, 3],
                        stride = 2,
                        padding = 'VALID',
                        scope = 'Conv2d_0a_3x3',
                    )
                net = tf.concat(axis = 3, values = [branch_0, branch_1])
                if add_and_check_final('Mixed_3a', net):
                    return net, end_points

            # 73 x 73 x 160
            with tf.variable_scope('Mixed_4a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(
                        net, 64, [1, 1], scope = 'Conv2d_0a_1x1'
                    )
                    branch_0 = slim.conv2d(
                        branch_0,
                        96,
                        [3, 3],
                        padding = 'VALID',
                        scope = 'Conv2d_1a_3x3',
                    )
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(
                        net, 64, [1, 1], scope = 'Conv2d_0a_1x1'
                    )
                    branch_1 = slim.conv2d(
                        branch_1, 64, [1, 7], scope = 'Conv2d_0b_1x7'
                    )
                    branch_1 = slim.conv2d(
                        branch_1, 64, [7, 1], scope = 'Conv2d_0c_7x1'
                    )
                    branch_1 = slim.conv2d(
                        branch_1,
                        96,
                        [3, 3],
                        padding = 'VALID',
                        scope = 'Conv2d_1a_3x3',
                    )
                net = tf.concat(axis = 3, values = [branch_0, branch_1])
                if add_and_check_final('Mixed_4a', net):
                    return net, end_points

            # 71 x 71 x 192
            with tf.variable_scope('Mixed_5a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(
                        net,
                        192,
                        [3, 3],
                        stride = 2,
                        padding = 'VALID',
                        scope = 'Conv2d_1a_3x3',
                    )
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.max_pool2d(
                        net,
                        [3, 3],
                        stride = 2,
                        padding = 'VALID',
                        scope = 'MaxPool_1a_3x3',
                    )
                net = tf.concat(axis = 3, values = [branch_0, branch_1])
                if add_and_check_final('Mixed_5a', net):
                    return net, end_points

            # 35 x 35 x 384
            # 4 x Inception-A blocks
            for idx in range(4):
                block_scope = 'Mixed_5' + chr(ord('b') + idx)
                net = block_inception_a(net, block_scope)
                if add_and_check_final(block_scope, net):
                    return net, end_points

            # 35 x 35 x 384
            # Reduction-A block
            net = block_reduction_a(net, 'Mixed_6a')
            if add_and_check_final('Mixed_6a', net):
                return net, end_points

            # 17 x 17 x 1024
            # 7 x Inception-B blocks
            for idx in range(7):
                block_scope = 'Mixed_6' + chr(ord('b') + idx)
                net = block_inception_b(net, block_scope)
                if add_and_check_final(block_scope, net):
                    return net, end_points

            # 17 x 17 x 1024
            # Reduction-B block
            net = block_reduction_b(net, 'Mixed_7a')
            if add_and_check_final('Mixed_7a', net):
                return net, end_points

            # 8 x 8 x 1536
            # 3 x Inception-C blocks
            for idx in range(3):
                block_scope = 'Mixed_7' + chr(ord('b') + idx)
                net = block_inception_c(net, block_scope)
                if add_and_check_final(block_scope, net):
                    return net, end_points
    raise ValueError('Unknown final endpoint %s' % final_endpoint)


from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Activation, Conv1D, Conv2D, Input, Lambda
from tensorflow.keras.layers import BatchNormalization, Flatten, Dense, Reshape
from tensorflow.keras.layers import (
    MaxPooling2D,
    AveragePooling2D,
    GlobalAveragePooling2D,
)
import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras.backend as K

weight_decay = 1e-4


class VladPooling(keras.layers.Layer):
    """
    This layer follows the NetVlad, GhostVlad
    """

    def __init__(self, mode, k_centers, g_centers = 0, **kwargs):
        self.k_centers = k_centers
        self.g_centers = g_centers
        self.mode = mode
        super(VladPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.cluster = self.add_weight(
            shape = [self.k_centers + self.g_centers, input_shape[0][-1]],
            name = 'centers',
            initializer = 'orthogonal',
        )
        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape
        return (input_shape[0][0], self.k_centers * input_shape[0][-1])

    def call(self, x):
        # feat : bz x W x H x D, cluster_score: bz X W x H x clusters.
        feat, cluster_score = x
        num_features = feat.shape[-1]

        # softmax normalization to get soft-assignment.
        # A : bz x W x H x clusters
        max_cluster_score = K.max(cluster_score, -1, keepdims = True)
        exp_cluster_score = K.exp(cluster_score - max_cluster_score)
        A = exp_cluster_score / K.sum(
            exp_cluster_score, axis = -1, keepdims = True
        )

        # Now, need to compute the residual, self.cluster: clusters x D
        A = K.expand_dims(A, -1)  # A : bz x W x H x clusters x 1
        feat_broadcast = K.expand_dims(
            feat, -2
        )  # feat_broadcast : bz x W x H x 1 x D
        feat_res = (
            feat_broadcast - self.cluster
        )  # feat_res : bz x W x H x clusters x D
        weighted_res = tf.multiply(
            A, feat_res
        )  # weighted_res : bz x W x H x clusters x D
        cluster_res = K.sum(weighted_res, [1, 2])

        if self.mode == 'gvlad':
            cluster_res = cluster_res[:, : self.k_centers, :]

        cluster_l2 = K.l2_normalize(cluster_res, -1)
        outputs = K.reshape(
            cluster_l2, [-1, int(self.k_centers) * int(num_features)]
        )
        return outputs


import librosa
import numpy as np


def load_wav(vid_path, sr = 16000, mode = 'eval'):
    wav, sr_ret = librosa.load(vid_path, sr = sr)
    assert sr_ret == sr
    if mode == 'train':
        extended_wav = np.append(wav, wav)
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
        return extended_wav
    else:
        extended_wav = np.append(wav, wav[::-1])
        return extended_wav


def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft = 1024):
    linear = librosa.stft(
        wav, n_fft = n_fft, win_length = win_length, hop_length = hop_length
    )
    return linear.T


def load_data(
    wav,
    win_length = 400,
    sr = 16000,
    hop_length = 160,
    n_fft = 512,
    spec_len = 120,
    mode = 'train',
):
    # wav = load_wav(path, sr=sr, mode=mode)
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    if mode == 'train':
        if time < spec_len:
            spec_mag = np.pad(mag_T, ((0, 0), (0, spec_len - time)), 'constant')
        else:
            spec_mag = mag_T
    else:
        spec_mag = mag_T
    mu = np.mean(spec_mag, 0, keepdims = True)
    std = np.std(spec_mag, 0, keepdims = True)
    return (spec_mag - mu) / (std + 1e-5)


def padding_sequence_nd(
    seq, maxlen = None, padding: str = 'post', pad_val = 0.0, dim: int = 1
):
    if padding not in ['post', 'pre']:
        raise ValueError('padding only supported [`post`, `pre`]')

    if not maxlen:
        maxlen = max([np.shape(s)[dim] for s in seq])

    padded_seqs = []
    for s in seq:
        npad = [[0, 0] for _ in range(len(s.shape))]
        if padding == 'pre':
            padding = 0
        if padding == 'post':
            padding = 1
        npad[dim][padding] = maxlen - s.shape[dim]
        padded_seqs.append(
            np.pad(
                s,
                pad_width = npad,
                mode = 'constant',
                constant_values = pad_val,
            )
        )
    return np.array(padded_seqs)


def add_noise(samples, noise, random_sample = True, factor = 0.1):
    y_noise = samples.copy()
    if len(y_noise) > len(noise):
        noise = np.tile(noise, int(np.ceil(len(y_noise) / len(noise))))
    else:
        if random_sample:
            noise = noise[np.random.randint(0, len(noise) - len(y_noise) + 1) :]
    return y_noise + noise[: len(y_noise)] * factor


def frames(audio, frame_duration_ms: int = 30, sample_rate: int = 16000):

    n = int(sample_rate * (frame_duration_ms / 1000.0))
    offset = 0
    timestamp = 0.0
    duration = float(n) / sample_rate
    results = []
    while offset + n < len(audio):
        results.append(audio[offset : offset + n])
        timestamp += duration
        offset += n
    return results


def random_sample(sample, sr, length = 500):
    sr = int(sr / 1000)
    r = np.random.randint(0, len(sample) - (sr * length))
    return sample[r : r + sr * length]


import pickle
import json

with open('../noise/noise.pkl', 'rb') as fopen:
    noises = pickle.load(fopen)

with open('../vggvox-speaker-identification/indices.json') as fopen:
    data = json.load(fopen)

files = data['files']
speakers = data['speakers']
unique_speakers = sorted(list(speakers.keys()))


def get_id(file):
    return file.split('/')[-1].split('-')[1]


unique_speakers.index(get_id(files[1]))

from sklearn.utils import shuffle
import itertools
import random

cycle_files = itertools.cycle(files)


# def generate(partition = 100, sample_rate = 16000, max_length = 5):
#     while True:
#         batch_files = [next(cycle_files) for _ in range(partition)]
#         X, Y = [], []
#         for file in batch_files:
#             y = unique_speakers.index(get_id(file))
#             w = load_wav(file)
#             if len(w) / sample_rate > max_length:
#                 X.append(w[: sample_rate * max_length])
#                 Y.append(y)
#             for _ in range(random.randint(1, 3)):
#                 f = frames(w, random.randint(500, max_length * 1000))
#                 X.extend(f)
#                 Y.extend([y] * len(f))

#         for k in range(len(X)):
#             if random.randint(0, 1):
#                 for _ in range(random.randint(1, 5)):
#                     x = add_noise(
#                         X[k], random.choice(noises), random.uniform(0.1, 0.6)
#                     )
#                     X.append(x)
#                     Y.append(Y[k])

#         actual_X, actual_Y = [], []

#         for k in range(len(X)):
#             try:
#                 actual_X.append(load_data(X[k]))
#                 actual_Y.append(Y[k])
#             except:
#                 pass

#         X, Y = shuffle(actual_X, actual_Y)

#         for k in range(len(X)):
#             yield {'inputs': np.expand_dims(X[k], -1), 'targets': [Y[k]]}


def generate(sample_rate = 16000, max_length = 5):
    while True:
        file = next(cycle_files)
        try:
            y = unique_speakers.index(get_id(file))
            w = load_wav(file)
            if len(w) / sample_rate > max_length:
                w = random_sample(
                    w, sample_rate, random.randint(500, max_length * 1000)
                )

            # if random.randint(0, 1):
            #     w = add_noise(
            #         w, random.choice(noises), random.uniform(0.1, 0.5)
            #     )
            w = load_data(w)
            yield {'inputs': np.expand_dims(w, -1), 'targets': [y]}
        except Exception as e:
            print(e)
            pass


import tensorflow as tf
import malaya_speech.train as train
import malaya_speech
from glob import glob


def get_dataset(batch_size = 32, shuffle_size = 5):
    def get():
        dataset = tf.data.Dataset.from_generator(
            generate,
            {'inputs': tf.float32, 'targets': tf.int32},
            output_shapes = {
                'inputs': tf.TensorShape([257, None, 1]),
                'targets': tf.TensorShape([1]),
            },
        )
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes = {
                'inputs': tf.TensorShape([257, None, 1]),
                'targets': tf.TensorShape([None]),
            },
            padding_values = {
                'inputs': tf.constant(0, dtype = tf.float32),
                'targets': tf.constant(0, dtype = tf.int32),
            },
        )
        dataset = dataset.shuffle(shuffle_size)
        return dataset

    return get


# def model(
#     inputs,
#     is_training = True,
#     dropout_keep_prob = 0.8,
#     reuse = None,
#     scope = 'InceptionV4',
# ):
#     with tf.variable_scope(
#         scope, 'InceptionV4', [inputs], reuse = reuse
#     ) as scope:
#         with slim.arg_scope(
#             [slim.batch_norm, slim.dropout], is_training = is_training
#         ):
#             net, end_points = inception_v4_base(inputs, scope = scope)
#             return net


# def model2(
#     inputs,
#     num_classes = 2,
#     is_training = True,
#     vlad_clusters = 8,
#     ghost_clusters = 2,
#     bottleneck_dim = 512,
# ):
#     with slim.arg_scope(inception_utils.inception_arg_scope()):
#         net = model(inputs, is_training = is_training)

#     x_fc = keras.layers.Conv2D(
#         bottleneck_dim,
#         (6, 1),
#         strides = (1, 1),
#         activation = 'relu',
#         kernel_initializer = 'orthogonal',
#         use_bias = True,
#         trainable = True,
#         kernel_regularizer = keras.regularizers.l2(weight_decay),
#         bias_regularizer = keras.regularizers.l2(weight_decay),
#         name = 'x_fc',
#     )(net)

#     x_k_center = keras.layers.Conv2D(
#         vlad_clusters + ghost_clusters,
#         (6, 1),
#         strides = (1, 1),
#         kernel_initializer = 'orthogonal',
#         use_bias = True,
#         trainable = True,
#         kernel_regularizer = keras.regularizers.l2(weight_decay),
#         bias_regularizer = keras.regularizers.l2(weight_decay),
#         name = 'gvlad_center_assignment',
#     )(net)
#     x = VladPooling(
#         k_centers = vlad_clusters,
#         g_centers = ghost_clusters,
#         mode = 'gvlad',
#         name = 'gvlad_pool',
#     )([x_fc, x_k_center])

#     x = keras.layers.Dense(
#         bottleneck_dim,
#         activation = 'relu',
#         kernel_initializer = 'orthogonal',
#         use_bias = True,
#         trainable = True,
#         kernel_regularizer = keras.regularizers.l2(weight_decay),
#         bias_regularizer = keras.regularizers.l2(weight_decay),
#         name = 'fc6',
#     )(x)

#     logits = keras.layers.Dense(
#         num_classes,
#         kernel_initializer = 'orthogonal',
#         use_bias = False,
#         trainable = True,
#         kernel_regularizer = keras.regularizers.l2(weight_decay),
#         bias_regularizer = keras.regularizers.l2(weight_decay),
#         name = 'prediction',
#     )(x)
#     return logits


def model(
    inputs,
    is_training = True,
    dropout_keep_prob = 0.8,
    reuse = None,
    scope = 'InceptionV4',
    num_classes = 2,
    bottleneck_dim = 512,
):
    # inputs = tf.image.grayscale_to_rgb(inputs)
    with tf.variable_scope(
        scope, 'InceptionV4', [inputs], reuse = reuse
    ) as scope:
        with slim.arg_scope(
            [slim.batch_norm, slim.dropout], is_training = is_training
        ):
            net, end_points = inception_v4_base(inputs, scope = scope)
            print(net.shape)

            with slim.arg_scope(
                [slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                stride = 1,
                padding = 'SAME',
            ):

                # Final pooling and prediction
                # TODO(sguada,arnoegw): Consider adding a parameter global_pool which
                # can be set to False to disable pooling here (as in resnet_*()).
                with tf.variable_scope('Logits'):
                    # 8 x 8 x 1536
                    kernel_size = net.get_shape()[1:3]
                    print(kernel_size)
                    if kernel_size.is_fully_defined():
                        net = slim.avg_pool2d(
                            net,
                            kernel_size,
                            padding = 'VALID',
                            scope = 'AvgPool_1a',
                        )
                    else:
                        net = tf.reduce_mean(
                            input_tensor = net,
                            axis = [1, 2],
                            keepdims = True,
                            name = 'global_pool',
                        )
                    end_points['global_pool'] = net
                    # 1 x 1 x 1536
                    net = slim.dropout(
                        net, dropout_keep_prob, scope = 'Dropout_1b'
                    )
                    net = slim.flatten(net, scope = 'PreLogitsFlatten')
                    end_points['PreLogitsFlatten'] = net
                    bottleneck = slim.fully_connected(
                        net, bottleneck_dim, scope = 'bottleneck'
                    )

                    # 1536
                    logits = slim.fully_connected(
                        bottleneck,
                        num_classes,
                        activation_fn = None,
                        scope = 'Logits',
                    )
                    return logits


init_lr = 1e-2
epochs = 500000


def model_fn(features, labels, mode, params):
    Y = features['targets'][:, 0]
    with slim.arg_scope(inception_utils.inception_arg_scope()):
        logits = model(features['inputs'], num_classes = len(unique_speakers))

    # logits = model2(features['inputs'], len(unique_speakers))

    # scale = 30
    # margin = 0.35
    # y_true = tf.one_hot(Y, len(unique_speakers))
    # y_pred = (y_true * (logits - margin) + (1 - y_true) * logits) * scale
    # cost = tf.nn.softmax_cross_entropy_with_logits(
    #     labels = y_true, logits = y_pred
    # )
    # loss = tf.reduce_mean(cost)

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

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.constant(
            value = init_lr, shape = [], dtype = tf.float32
        )
        learning_rate = tf.train.polynomial_decay(
            learning_rate,
            global_step,
            epochs,
            end_learning_rate = 0.00001,
            power = 1.0,
            cycle = False,
        )
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate, decay = 0.9, momentum = 0.9, epsilon = 1.0
        )
        # optimizer = tf.train.AdamOptimizer(learning_rate)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)

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
train_dataset = get_dataset(batch_size = 32)

save_directory = 'output-inception-v4'

train.run_training(
    train_fn = train_dataset,
    model_fn = model_fn,
    model_dir = save_directory,
    num_gpus = 3,
    log_step = 1,
    save_checkpoint_step = 25000,
    max_steps = epochs,
    train_hooks = train_hooks,
)
