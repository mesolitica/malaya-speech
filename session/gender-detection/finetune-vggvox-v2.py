import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../gcs/mesolitica-storage.json'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf

import collections
import re


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


def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(
        wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length
    )  # linear spectrogram
    return linear.T


def load_data(
    wav,
    win_length=400,
    sr=16000,
    hop_length=160,
    n_fft=512,
    spec_len=100,
    mode='train',
):
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
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    return (spec_mag - mu) / (std + 1e-5)


n_mels = 257


def calc(v):

    r = load_data(v, mode='train')
    return r


def preprocess_inputs(example):
    s = tf.compat.v1.numpy_function(calc, [example['inputs']], tf.float32)

    s = tf.reshape(s, (n_mels, -1, 1))
    example['inputs'] = s

    return example


def parse(serialized_example):

    data_fields = {
        'inputs': tf.VarLenFeature(tf.float32),
        'targets': tf.VarLenFeature(tf.int64),
    }
    features = tf.parse_single_example(
        serialized_example, features=data_fields
    )
    for k in features.keys():
        features[k] = features[k].values

    features = preprocess_inputs(features)

    keys = list(features.keys())
    for k in keys:
        if k not in ['inputs', 'targets']:
            features.pop(k, None)

    return features


def get_dataset(files, batch_size=32, shuffle_size=1024, thread_count=24):
    def get():
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.map(parse, num_parallel_calls=thread_count)
        dataset = dataset.shuffle(shuffle_size)
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes={
                'inputs': tf.TensorShape([n_mels, None, 1]),
                'targets': tf.TensorShape([None]),
            },
            padding_values={
                'inputs': tf.constant(0, dtype=tf.float32),
                'targets': tf.constant(0, dtype=tf.int64),
            },
        )
        dataset = dataset.repeat()
        return dataset

    return get


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
    input_tensor, kernel_size, filters, stage, block, trainable=True
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
        kernel_initializer='orthogonal',
        use_bias=False,
        trainable=trainable,
        kernel_regularizer=l2(weight_decay),
        name=conv_name_1,
    )(input_tensor)
    x = BatchNormalization(
        axis=bn_axis, trainable=trainable, name=bn_name_1
    )(x)
    x = Activation('relu')(x)

    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
    bn_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3/bn'
    x = Conv2D(
        filters2,
        kernel_size,
        padding='same',
        kernel_initializer='orthogonal',
        use_bias=False,
        trainable=trainable,
        kernel_regularizer=l2(weight_decay),
        name=conv_name_2,
    )(x)
    x = BatchNormalization(
        axis=bn_axis, trainable=trainable, name=bn_name_2
    )(x)
    x = Activation('relu')(x)

    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase/bn'
    x = Conv2D(
        filters3,
        (1, 1),
        kernel_initializer='orthogonal',
        use_bias=False,
        trainable=trainable,
        kernel_regularizer=l2(weight_decay),
        name=conv_name_3,
    )(x)
    x = BatchNormalization(
        axis=bn_axis, trainable=trainable, name=bn_name_3
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
    strides=(2, 2),
    trainable=True,
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
        strides=strides,
        kernel_initializer='orthogonal',
        use_bias=False,
        trainable=trainable,
        kernel_regularizer=l2(weight_decay),
        name=conv_name_1,
    )(input_tensor)
    x = BatchNormalization(
        axis=bn_axis, trainable=trainable, name=bn_name_1
    )(x)
    x = Activation('relu')(x)

    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
    bn_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3/bn'
    x = Conv2D(
        filters2,
        kernel_size,
        padding='same',
        kernel_initializer='orthogonal',
        use_bias=False,
        trainable=trainable,
        kernel_regularizer=l2(weight_decay),
        name=conv_name_2,
    )(x)
    x = BatchNormalization(
        axis=bn_axis, trainable=trainable, name=bn_name_2
    )(x)
    x = Activation('relu')(x)

    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase/bn'
    x = Conv2D(
        filters3,
        (1, 1),
        kernel_initializer='orthogonal',
        use_bias=False,
        trainable=trainable,
        kernel_regularizer=l2(weight_decay),
        name=conv_name_3,
    )(x)
    x = BatchNormalization(
        axis=bn_axis, trainable=trainable, name=bn_name_3
    )(x)

    conv_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_proj'
    bn_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_proj/bn'
    shortcut = Conv2D(
        filters3,
        (1, 1),
        strides=strides,
        kernel_initializer='orthogonal',
        use_bias=False,
        trainable=trainable,
        kernel_regularizer=l2(weight_decay),
        name=conv_name_4,
    )(input_tensor)
    shortcut = BatchNormalization(
        axis=bn_axis, trainable=trainable, name=bn_name_4
    )(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet_2D_v1(inputs, mode='train'):
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
        kernel_initializer='orthogonal',
        use_bias=False,
        trainable=True,
        kernel_regularizer=l2(weight_decay),
        padding='same',
        name='conv1_1/3x3_s1',
    )(inputs)

    x1 = BatchNormalization(
        axis=bn_axis, name='conv1_1/3x3_s1/bn', trainable=True
    )(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2))(x1)

    # ===============================================
    #            Convolution Section 2
    # ===============================================
    x2 = conv_block_2D(
        x1,
        3,
        [48, 48, 96],
        stage=2,
        block='a',
        strides=(1, 1),
        trainable=True,
    )
    x2 = identity_block_2D(
        x2, 3, [48, 48, 96], stage=2, block='b', trainable=True
    )

    # ===============================================
    #            Convolution Section 3
    # ===============================================
    x3 = conv_block_2D(
        x2, 3, [96, 96, 128], stage=3, block='a', trainable=True
    )
    x3 = identity_block_2D(
        x3, 3, [96, 96, 128], stage=3, block='b', trainable=True
    )
    x3 = identity_block_2D(
        x3, 3, [96, 96, 128], stage=3, block='c', trainable=True
    )
    # ===============================================
    #            Convolution Section 4
    # ===============================================
    x4 = conv_block_2D(
        x3, 3, [128, 128, 256], stage=4, block='a', trainable=True
    )
    x4 = identity_block_2D(
        x4, 3, [128, 128, 256], stage=4, block='b', trainable=True
    )
    x4 = identity_block_2D(
        x4, 3, [128, 128, 256], stage=4, block='c', trainable=True
    )
    # ===============================================
    #            Convolution Section 5
    # ===============================================
    x5 = conv_block_2D(
        x4, 3, [256, 256, 512], stage=5, block='a', trainable=True
    )
    x5 = identity_block_2D(
        x5, 3, [256, 256, 512], stage=5, block='b', trainable=True
    )
    x5 = identity_block_2D(
        x5, 3, [256, 256, 512], stage=5, block='c', trainable=True
    )
    y = MaxPooling2D((3, 1), strides=(2, 1), name='mpool2')(x5)
    return inputs, y


def resnet_2D_v2(inputs, mode='train'):
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
        strides=(2, 2),
        kernel_initializer='orthogonal',
        use_bias=False,
        trainable=True,
        kernel_regularizer=l2(weight_decay),
        padding='same',
        name='conv1_1/3x3_s1',
    )(inputs)

    x1 = BatchNormalization(
        axis=bn_axis, name='conv1_1/3x3_s1/bn', trainable=True
    )(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2))(x1)

    # ===============================================
    #            Convolution Section 2
    # ===============================================
    x2 = conv_block_2D(
        x1,
        3,
        [64, 64, 256],
        stage=2,
        block='a',
        strides=(1, 1),
        trainable=True,
    )
    x2 = identity_block_2D(
        x2, 3, [64, 64, 256], stage=2, block='b', trainable=True
    )
    x2 = identity_block_2D(
        x2, 3, [64, 64, 256], stage=2, block='c', trainable=True
    )
    # ===============================================
    #            Convolution Section 3
    # ===============================================
    x3 = conv_block_2D(
        x2, 3, [128, 128, 512], stage=3, block='a', trainable=True
    )
    x3 = identity_block_2D(
        x3, 3, [128, 128, 512], stage=3, block='b', trainable=True
    )
    x3 = identity_block_2D(
        x3, 3, [128, 128, 512], stage=3, block='c', trainable=True
    )
    # ===============================================
    #            Convolution Section 4
    # ===============================================
    x4 = conv_block_2D(
        x3,
        3,
        [256, 256, 1024],
        stage=4,
        block='a',
        strides=(1, 1),
        trainable=True,
    )
    x4 = identity_block_2D(
        x4, 3, [256, 256, 1024], stage=4, block='b', trainable=True
    )
    x4 = identity_block_2D(
        x4, 3, [256, 256, 1024], stage=4, block='c', trainable=True
    )
    # ===============================================
    #            Convolution Section 5
    # ===============================================
    x5 = conv_block_2D(
        x4, 3, [512, 512, 2048], stage=5, block='a', trainable=True
    )
    x5 = identity_block_2D(
        x5, 3, [512, 512, 2048], stage=5, block='b', trainable=True
    )
    x5 = identity_block_2D(
        x5, 3, [512, 512, 2048], stage=5, block='c', trainable=True
    )
    y = MaxPooling2D((3, 1), strides=(2, 1), name='mpool2')(x5)
    return inputs, y


class VladPooling(keras.layers.Layer):
    """
    This layer follows the NetVlad, GhostVlad
    """

    def __init__(self, mode, k_centers, g_centers=0, **kwargs):
        self.k_centers = k_centers
        self.g_centers = g_centers
        self.mode = mode
        super(VladPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.cluster = self.add_weight(
            shape=[self.k_centers + self.g_centers, input_shape[0][-1]],
            name='centers',
            initializer='orthogonal',
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
        max_cluster_score = K.max(cluster_score, -1, keepdims=True)
        exp_cluster_score = K.exp(cluster_score - max_cluster_score)
        A = exp_cluster_score / K.sum(
            exp_cluster_score, axis=-1, keepdims=True
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


def amsoftmax_loss(y_true, y_pred, scale=30, margin=0.35):
    y_pred = y_true * (y_pred - margin) + (1 - y_true) * y_pred
    y_pred *= scale
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)


def vggvox_resnet2d_icassp(
    inputs, num_class=8631, mode='train', args=None
):

    net = 'resnet34s'
    loss = 'softmax'
    vlad_clusters = 8
    ghost_clusters = 2
    bottleneck_dim = 512
    aggregation = 'gvlad'
    mgpu = 0

    if net == 'resnet34s':
        inputs, x = resnet_2D_v1(inputs, mode=mode)
    else:
        inputs, x = resnet_2D_v2(inputs, mode=mode)

    x_fc = keras.layers.Conv2D(
        bottleneck_dim,
        (7, 1),
        strides=(1, 1),
        activation='relu',
        kernel_initializer='orthogonal',
        use_bias=True,
        trainable=True,
        kernel_regularizer=keras.regularizers.l2(weight_decay),
        bias_regularizer=keras.regularizers.l2(weight_decay),
        name='x_fc',
    )(x)

    # ===============================================
    #            Feature Aggregation
    # ===============================================
    if aggregation == 'avg':
        if mode == 'train':
            x = keras.layers.AveragePooling2D(
                (1, 5), strides=(1, 1), name='avg_pool'
            )(x)
            x = keras.layers.Reshape((-1, bottleneck_dim))(x)
        else:
            x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
            x = keras.layers.Reshape((1, bottleneck_dim))(x)

    elif aggregation == 'vlad':
        x_k_center = keras.layers.Conv2D(
            vlad_clusters,
            (7, 1),
            strides=(1, 1),
            kernel_initializer='orthogonal',
            use_bias=True,
            trainable=True,
            kernel_regularizer=keras.regularizers.l2(weight_decay),
            bias_regularizer=keras.regularizers.l2(weight_decay),
            name='vlad_center_assignment',
        )(x)
        x = VladPooling(
            k_centers=vlad_clusters, mode='vlad', name='vlad_pool'
        )([x_fc, x_k_center])

    elif aggregation == 'gvlad':
        x_k_center = keras.layers.Conv2D(
            vlad_clusters + ghost_clusters,
            (7, 1),
            strides=(1, 1),
            kernel_initializer='orthogonal',
            use_bias=True,
            trainable=True,
            kernel_regularizer=keras.regularizers.l2(weight_decay),
            bias_regularizer=keras.regularizers.l2(weight_decay),
            name='gvlad_center_assignment',
        )(x)
        x = VladPooling(
            k_centers=vlad_clusters,
            g_centers=ghost_clusters,
            mode='gvlad',
            name='gvlad_pool',
        )([x_fc, x_k_center])

    else:
        raise IOError('==> unknown aggregation mode')
    x = keras.layers.Dense(
        bottleneck_dim,
        activation='relu',
        kernel_initializer='orthogonal',
        use_bias=True,
        trainable=True,
        kernel_regularizer=keras.regularizers.l2(weight_decay),
        bias_regularizer=keras.regularizers.l2(weight_decay),
        name='fc6',
    )(x)
    if loss == 'softmax':
        y = keras.layers.Dense(
            num_class,
            activation='softmax',
            kernel_initializer='orthogonal',
            use_bias=False,
            trainable=True,
            kernel_regularizer=keras.regularizers.l2(weight_decay),
            bias_regularizer=keras.regularizers.l2(weight_decay),
            name='prediction',
        )(x)
        trnloss = 'categorical_crossentropy'

    elif loss == 'amsoftmax':
        x_l2 = keras.layers.Lambda(lambda x: K.l2_normalize(x, 1))(x)
        y = keras.layers.Dense(
            num_class,
            kernel_initializer='orthogonal',
            use_bias=False,
            trainable=True,
            kernel_constraint=keras.constraints.unit_norm(),
            kernel_regularizer=keras.regularizers.l2(weight_decay),
            bias_regularizer=keras.regularizers.l2(weight_decay),
            name='prediction',
        )(x_l2)
        trnloss = amsoftmax_loss

    else:
        raise IOError('==> unknown loss.')

    if mode == 'eval':
        y = keras.layers.Lambda(lambda x: keras.backend.l2_normalize(x, 1))(x)

    return y


learning_rate = 1e-5
init_checkpoint = '../vggvox-speaker-identification/v2/vggvox.ckpt'


def model_fn(features, labels, mode, params):
    Y = tf.cast(features['targets'][:, 0], tf.int32)

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
    logits = vggvox_resnet2d_icassp(
        features['inputs'], num_class=3, mode='train'
    )

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=Y
        )
    )

    tf.identity(loss, 'train_loss')

    accuracy = tf.metrics.accuracy(
        labels=Y, predictions=tf.argmax(logits, axis=1)
    )

    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    variables = [v for v in variables if 'prediction' not in v.name]

    assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(
        variables, init_checkpoint
    )

    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        estimator_spec = tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op
        )

    elif mode == tf.estimator.ModeKeys.EVAL:

        estimator_spec = tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={'accuracy': accuracy},
        )

    return estimator_spec


train_hooks = [
    tf.train.LoggingTensorHook(
        ['train_accuracy', 'train_loss'], every_n_iter=1
    )
]

files = tf.io.gfile.glob('gs://mesolitica-general/gender/data/*.tfrecords')
train_dataset = get_dataset(files)

save_directory = 'output-vggvox-v2-gender'

train.run_training(
    train_fn=train_dataset,
    model_fn=model_fn,
    model_dir=save_directory,
    num_gpus=1,
    log_step=1,
    save_checkpoint_step=25000,
    max_steps=300000,
    train_hooks=train_hooks,
)
