import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from glob import glob
import random

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


files = glob('../voxceleb/wav/*.wav')
random.shuffle(files)
len(files)

import pickle

with open('../noise/noise.pkl', 'rb') as fopen:
    noises = pickle.load(fopen)


def get_id(file):
    return file.split('/')[-1].split('-')[1]


from collections import defaultdict

speakers = defaultdict(list)

for no, file in enumerate(files):
    speaker = get_id(file)
    speakers[speaker].append(no)

unique_speakers = sorted(list(speakers.keys()))

import json

with open('indices.json', 'w') as fopen:
    json.dump({'files': files, 'speakers': speakers}, fopen)

from sklearn.utils import shuffle
import itertools

cycle_files = itertools.cycle(files)


def generate(
    partition = 100, batch_size = 32, sample_rate = 16000, max_length = 5
):
    while True:
        batch_files = [next(cycle_files) for _ in range(partition)]
        X, Y = [], []
        for file in batch_files:
            y = unique_speakers.index(get_id(file))
            w = load_wav(file)
            if len(w) / sample_rate > max_length:
                X.append(w[: sample_rate * max_length])
                Y.append(y)
            for _ in range(random.randint(1, 3)):
                f = frames(w, random.randint(500, max_length * 1000))
                X.extend(f)
                Y.extend([y] * len(f))

        for k in range(len(X)):
            if random.randint(0, 1):
                for _ in range(random.randint(1, 5)):
                    x = add_noise(
                        X[k], random.choice(noises), random.uniform(0.1, 0.6)
                    )
                    X.append(x)
                    Y.append(Y[k])

        actual_X, actual_Y = [], []

        for k in range(len(X)):
            try:
                actual_X.append(load_data(X[k]))
                actual_Y.append(Y[k])
            except:
                pass

        X, Y = shuffle(actual_X, actual_Y)

        for k in range(0, len(X), batch_size):
            batch_x = X[k : k + batch_size]
            batch_y = Y[k : k + batch_size]
            yield padding_sequence_nd(batch_x), batch_y


g = generate()

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


def identity_block_2D(
    input_tensor, kernel_size, filters, stage, block, trainable = True
):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
    bn_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce/bn'
    x = Conv2D(
        filters1,
        (1, 1),
        kernel_initializer = 'orthogonal',
        use_bias = False,
        trainable = trainable,
        kernel_regularizer = l2(weight_decay),
        name = conv_name_1,
    )(input_tensor)
    x = BatchNormalization(
        axis = bn_axis, trainable = trainable, name = bn_name_1
    )(x)
    x = Activation('relu')(x)

    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
    bn_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3/bn'
    x = Conv2D(
        filters2,
        kernel_size,
        padding = 'same',
        kernel_initializer = 'orthogonal',
        use_bias = False,
        trainable = trainable,
        kernel_regularizer = l2(weight_decay),
        name = conv_name_2,
    )(x)
    x = BatchNormalization(
        axis = bn_axis, trainable = trainable, name = bn_name_2
    )(x)
    x = Activation('relu')(x)

    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase/bn'
    x = Conv2D(
        filters3,
        (1, 1),
        kernel_initializer = 'orthogonal',
        use_bias = False,
        trainable = trainable,
        kernel_regularizer = l2(weight_decay),
        name = conv_name_3,
    )(x)
    x = BatchNormalization(
        axis = bn_axis, trainable = trainable, name = bn_name_3
    )(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block_2D(
    input_tensor,
    kernel_size,
    filters,
    stage,
    block,
    strides = (2, 2),
    trainable = True,
):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
    bn_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce/bn'
    x = Conv2D(
        filters1,
        (1, 1),
        strides = strides,
        kernel_initializer = 'orthogonal',
        use_bias = False,
        trainable = trainable,
        kernel_regularizer = l2(weight_decay),
        name = conv_name_1,
    )(input_tensor)
    x = BatchNormalization(
        axis = bn_axis, trainable = trainable, name = bn_name_1
    )(x)
    x = Activation('relu')(x)

    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
    bn_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3/bn'
    x = Conv2D(
        filters2,
        kernel_size,
        padding = 'same',
        kernel_initializer = 'orthogonal',
        use_bias = False,
        trainable = trainable,
        kernel_regularizer = l2(weight_decay),
        name = conv_name_2,
    )(x)
    x = BatchNormalization(
        axis = bn_axis, trainable = trainable, name = bn_name_2
    )(x)
    x = Activation('relu')(x)

    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase/bn'
    x = Conv2D(
        filters3,
        (1, 1),
        kernel_initializer = 'orthogonal',
        use_bias = False,
        trainable = trainable,
        kernel_regularizer = l2(weight_decay),
        name = conv_name_3,
    )(x)
    x = BatchNormalization(
        axis = bn_axis, trainable = trainable, name = bn_name_3
    )(x)

    conv_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_proj'
    bn_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_proj/bn'
    shortcut = Conv2D(
        filters3,
        (1, 1),
        strides = strides,
        kernel_initializer = 'orthogonal',
        use_bias = False,
        trainable = trainable,
        kernel_regularizer = l2(weight_decay),
        name = conv_name_4,
    )(input_tensor)
    shortcut = BatchNormalization(
        axis = bn_axis, trainable = trainable, name = bn_name_4
    )(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet_2D_v1(inputs, mode = 'train'):
    bn_axis = 3
    #     if mode == 'train':
    #         inputs = Input(shape=input_dim, name='input')
    #     else:
    #         inputs = Input(shape=(input_dim[0], None, input_dim[-1]), name='input')
    # ===============================================
    #            Convolution Block 1
    # ===============================================
    x1 = Conv2D(
        64,
        (7, 7),
        kernel_initializer = 'orthogonal',
        use_bias = False,
        trainable = True,
        kernel_regularizer = l2(weight_decay),
        padding = 'same',
        name = 'conv1_1/3x3_s1',
    )(inputs)

    x1 = BatchNormalization(
        axis = bn_axis, name = 'conv1_1/3x3_s1/bn', trainable = True
    )(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D((2, 2), strides = (2, 2))(x1)

    # ===============================================
    #            Convolution Section 2
    # ===============================================
    x2 = conv_block_2D(
        x1,
        3,
        [48, 48, 96],
        stage = 2,
        block = 'a',
        strides = (1, 1),
        trainable = True,
    )
    x2 = identity_block_2D(
        x2, 3, [48, 48, 96], stage = 2, block = 'b', trainable = True
    )

    # ===============================================
    #            Convolution Section 3
    # ===============================================
    x3 = conv_block_2D(
        x2, 3, [96, 96, 128], stage = 3, block = 'a', trainable = True
    )
    x3 = identity_block_2D(
        x3, 3, [96, 96, 128], stage = 3, block = 'b', trainable = True
    )
    x3 = identity_block_2D(
        x3, 3, [96, 96, 128], stage = 3, block = 'c', trainable = True
    )
    # ===============================================
    #            Convolution Section 4
    # ===============================================
    x4 = conv_block_2D(
        x3, 3, [128, 128, 256], stage = 4, block = 'a', trainable = True
    )
    x4 = identity_block_2D(
        x4, 3, [128, 128, 256], stage = 4, block = 'b', trainable = True
    )
    x4 = identity_block_2D(
        x4, 3, [128, 128, 256], stage = 4, block = 'c', trainable = True
    )
    # ===============================================
    #            Convolution Section 5
    # ===============================================
    x5 = conv_block_2D(
        x4, 3, [256, 256, 512], stage = 5, block = 'a', trainable = True
    )
    x5 = identity_block_2D(
        x5, 3, [256, 256, 512], stage = 5, block = 'b', trainable = True
    )
    x5 = identity_block_2D(
        x5, 3, [256, 256, 512], stage = 5, block = 'c', trainable = True
    )
    y = MaxPooling2D((3, 1), strides = (2, 1), name = 'mpool2')(x5)
    return inputs, y


def resnet_2D_v2(inputs, mode = 'train'):
    bn_axis = 3
    #     if mode == 'train':
    #         inputs = Input(shape=input_dim, name='input')
    #     else:
    #         inputs = Input(shape=(input_dim[0], None, input_dim[-1]), name='input')
    # ===============================================
    #            Convolution Block 1
    # ===============================================
    x1 = Conv2D(
        64,
        (7, 7),
        strides = (2, 2),
        kernel_initializer = 'orthogonal',
        use_bias = False,
        trainable = True,
        kernel_regularizer = l2(weight_decay),
        padding = 'same',
        name = 'conv1_1/3x3_s1',
    )(inputs)

    x1 = BatchNormalization(
        axis = bn_axis, name = 'conv1_1/3x3_s1/bn', trainable = True
    )(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D((2, 2), strides = (2, 2))(x1)

    # ===============================================
    #            Convolution Section 2
    # ===============================================
    x2 = conv_block_2D(
        x1,
        3,
        [64, 64, 256],
        stage = 2,
        block = 'a',
        strides = (1, 1),
        trainable = True,
    )
    x2 = identity_block_2D(
        x2, 3, [64, 64, 256], stage = 2, block = 'b', trainable = True
    )
    x2 = identity_block_2D(
        x2, 3, [64, 64, 256], stage = 2, block = 'c', trainable = True
    )
    # ===============================================
    #            Convolution Section 3
    # ===============================================
    x3 = conv_block_2D(
        x2, 3, [128, 128, 512], stage = 3, block = 'a', trainable = True
    )
    x3 = identity_block_2D(
        x3, 3, [128, 128, 512], stage = 3, block = 'b', trainable = True
    )
    x3 = identity_block_2D(
        x3, 3, [128, 128, 512], stage = 3, block = 'c', trainable = True
    )
    # ===============================================
    #            Convolution Section 4
    # ===============================================
    x4 = conv_block_2D(
        x3,
        3,
        [256, 256, 1024],
        stage = 4,
        block = 'a',
        strides = (1, 1),
        trainable = True,
    )
    x4 = identity_block_2D(
        x4, 3, [256, 256, 1024], stage = 4, block = 'b', trainable = True
    )
    x4 = identity_block_2D(
        x4, 3, [256, 256, 1024], stage = 4, block = 'c', trainable = True
    )
    # ===============================================
    #            Convolution Section 5
    # ===============================================
    x5 = conv_block_2D(
        x4, 3, [512, 512, 2048], stage = 5, block = 'a', trainable = True
    )
    x5 = identity_block_2D(
        x5, 3, [512, 512, 2048], stage = 5, block = 'b', trainable = True
    )
    x5 = identity_block_2D(
        x5, 3, [512, 512, 2048], stage = 5, block = 'c', trainable = True
    )
    y = MaxPooling2D((3, 1), strides = (2, 1), name = 'mpool2')(x5)
    return inputs, y


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


def vggvox_resnet2d_icassp(
    inputs, num_class = 8631, mode = 'train', args = None
):

    # python predict.py --gpu 1 --net resnet34s --ghost_cluster 2
    # --vlad_cluster 8 --loss softmax --resume

    net = 'resnet34s'
    loss = 'softmax'
    vlad_clusters = 8
    ghost_clusters = 2
    bottleneck_dim = 512
    aggregation = 'gvlad'
    mgpu = 0

    if net == 'resnet34s':
        inputs, x = resnet_2D_v1(inputs, mode = mode)
    else:
        inputs, x = resnet_2D_v2(inputs, mode = mode)
    # ===============================================
    #            Fully Connected Block 1
    # ===============================================
    x_fc = keras.layers.Conv2D(
        bottleneck_dim,
        (7, 1),
        strides = (1, 1),
        activation = 'relu',
        kernel_initializer = 'orthogonal',
        use_bias = True,
        trainable = True,
        kernel_regularizer = keras.regularizers.l2(weight_decay),
        bias_regularizer = keras.regularizers.l2(weight_decay),
        name = 'x_fc',
    )(x)

    # ===============================================
    #            Feature Aggregation
    # ===============================================
    if aggregation == 'avg':
        if mode == 'train':
            x = keras.layers.AveragePooling2D(
                (1, 5), strides = (1, 1), name = 'avg_pool'
            )(x)
            x = keras.layers.Reshape((-1, bottleneck_dim))(x)
        else:
            x = keras.layers.GlobalAveragePooling2D(name = 'avg_pool')(x)
            x = keras.layers.Reshape((1, bottleneck_dim))(x)

    elif aggregation == 'vlad':
        x_k_center = keras.layers.Conv2D(
            vlad_clusters,
            (7, 1),
            strides = (1, 1),
            kernel_initializer = 'orthogonal',
            use_bias = True,
            trainable = True,
            kernel_regularizer = keras.regularizers.l2(weight_decay),
            bias_regularizer = keras.regularizers.l2(weight_decay),
            name = 'vlad_center_assignment',
        )(x)
        x = VladPooling(
            k_centers = vlad_clusters, mode = 'vlad', name = 'vlad_pool'
        )([x_fc, x_k_center])

    elif aggregation == 'gvlad':
        x_k_center = keras.layers.Conv2D(
            vlad_clusters + ghost_clusters,
            (7, 1),
            strides = (1, 1),
            kernel_initializer = 'orthogonal',
            use_bias = True,
            trainable = True,
            kernel_regularizer = keras.regularizers.l2(weight_decay),
            bias_regularizer = keras.regularizers.l2(weight_decay),
            name = 'gvlad_center_assignment',
        )(x)
        x = VladPooling(
            k_centers = vlad_clusters,
            g_centers = ghost_clusters,
            mode = 'gvlad',
            name = 'gvlad_pool',
        )([x_fc, x_k_center])

    else:
        raise IOError('==> unknown aggregation mode')

    # ===============================================
    #            Fully Connected Block 2
    # ===============================================
    x = keras.layers.Dense(
        bottleneck_dim,
        activation = 'relu',
        kernel_initializer = 'orthogonal',
        use_bias = True,
        trainable = True,
        kernel_regularizer = keras.regularizers.l2(weight_decay),
        bias_regularizer = keras.regularizers.l2(weight_decay),
        name = 'fc6',
    )(x)

    x_l2 = keras.layers.Lambda(lambda x: K.l2_normalize(x, 1))(x)
    y = keras.layers.Dense(
        num_class,
        kernel_initializer = 'orthogonal',
        use_bias = False,
        trainable = True,
        kernel_constraint = keras.constraints.unit_norm(),
        kernel_regularizer = keras.regularizers.l2(weight_decay),
        bias_regularizer = keras.regularizers.l2(weight_decay),
        name = 'prediction',
    )(x_l2)

    if mode == 'eval':
        y = keras.layers.Lambda(lambda x: keras.backend.l2_normalize(x, 1))(x)

    return y


class Model:
    def __init__(self, epochs, num_warmup_steps = 10000, init_lr = 1e-3):
        self.X = tf.placeholder(tf.float32, [None, 257, None, 1])
        self.Y = tf.placeholder(tf.int32, [None, None])
        self.batch_size = tf.shape(self.X)[0]

        params = {
            'dim': (257, None, 1),
            'nfft': 512,
            'spec_len': 250,
            'win_length': 400,
            'hop_length': 160,
            'n_classes': 5994,
            'sampling_rate': 16000,
            'normalize': True,
        }
        self.logits = vggvox_resnet2d_icassp(
            self.X, num_class = len(unique_speakers), mode = 'train'
        )
        self.logits = tf.identity(self.logits, name = 'logits')

        self.gamma = 64
        self.margin = 0.25
        self.O_p = 1 + self.margin
        self.O_n = -self.margin
        self.Delta_p = 1 - self.margin
        self.Delta_n = self.margin

        self.batch_idxs = tf.expand_dims(
            tf.range(0, self.batch_size, dtype = tf.int32), 1
        )  # shape [batch,1]
        idxs = tf.concat([self.batch_idxs, tf.cast(self.Y, tf.int32)], 1)
        sp = tf.expand_dims(tf.gather_nd(self.logits, idxs), 1)
        mask = tf.logical_not(
            tf.scatter_nd(
                idxs, tf.ones(tf.shape(idxs)[0], tf.bool), tf.shape(self.logits)
            )
        )

        sn = tf.reshape(
            tf.boolean_mask(self.logits, mask), (self.batch_size, -1)
        )

        alpha_p = tf.nn.relu(self.O_p - tf.stop_gradient(sp))
        alpha_n = tf.nn.relu(tf.stop_gradient(sn) - self.O_n)

        r_sp_m = alpha_p * (sp - self.Delta_p)
        r_sn_m = alpha_n * (sn - self.Delta_n)
        _Z = tf.concat([r_sn_m, r_sp_m], 1)
        _Z = _Z * self.gamma
        # sum all similarity
        logZ = tf.math.reduce_logsumexp(_Z, 1, keepdims = True)
        # remove sn_p from all sum similarity
        self.cost = -r_sp_m * self.gamma + logZ
        self.cost = tf.reduce_mean(self.cost[:, 0])

        global_step = tf.train.get_or_create_global_step()
        self.global_step = global_step

        learning_rate = tf.constant(
            value = init_lr, shape = [], dtype = tf.float32
        )

        learning_rate = tf.train.polynomial_decay(
            learning_rate,
            global_step,
            epochs,
            end_learning_rate = 0.0,
            power = 1.0,
            cycle = False,
        )

        if num_warmup_steps:
            global_steps_int = tf.cast(global_step, tf.int32)
            warmup_steps_int = tf.constant(num_warmup_steps, dtype = tf.int32)

            global_steps_float = tf.cast(global_steps_int, tf.float32)
            warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

            warmup_percent_done = global_steps_float / warmup_steps_float
            warmup_learning_rate = init_lr * warmup_percent_done

            is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
            learning_rate = (
                1.0 - is_warmup
            ) * learning_rate + is_warmup * warmup_learning_rate

        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
            self.cost, global_step = global_step
        )

        correct_pred = tf.equal(
            tf.argmax(self.logits, 1, output_type = tf.int32), self.Y[:, 0]
        )
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        cost_summary = tf.summary.scalar('cost', self.cost)
        accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()


max_steps = 500_000
save_steps = 25000

tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(epochs = max_steps)
sess.run(tf.global_variables_initializer())

directory = 'vggvox-v2-circleloss'

writer = tf.summary.FileWriter(f'./{directory}')
saver = tf.train.Saver()

latest_checkpoint = tf.train.latest_checkpoint(directory)
if latest_checkpoint:
    saver.restore(sess, latest_checkpoint)

from tqdm import tqdm

pbar = tqdm(range(max_steps), desc = 'train minibatch loop')
for i in pbar:

    if (i + 1) % save_steps == 0:
        saver.save(sess, f'{directory}/model.ckpt', global_step = i)

    x, y = next(g)
    x = np.expand_dims(x, -1)
    y = np.expand_dims(y, -1)

    acc, cost, summary, _ = sess.run(
        [model.accuracy, model.cost, model.merged, model.optimizer],
        feed_dict = {model.Y: y, model.X: x},
    )

    writer.add_summary(summary, i)

    pbar.set_postfix(cost = cost, accuracy = acc)

saver.save(sess, f'{directory}/model.ckpt', global_step = max_steps)
