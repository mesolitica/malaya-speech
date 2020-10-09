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


def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft = 1024):
    linear = librosa.stft(
        wav, n_fft = n_fft, win_length = win_length, hop_length = hop_length
    )  # linear spectrogram
    return linear.T


def load_data(
    wav,
    win_length = 400,
    sr = 16000,
    hop_length = 24,
    n_fft = 512,
    spec_len = 100,
    mode = 'train',
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
    mu = np.mean(spec_mag, 0, keepdims = True)
    std = np.std(spec_mag, 0, keepdims = True)
    return (spec_mag - mu) / (std + 1e-5)


n_mels = 257


def calc(v):

    r = load_data(v, mode = 'train')
    return r


def preprocess_inputs(example):
    s = tf.compat.v1.numpy_function(calc, [example['waveforms']], tf.float32)

    s = tf.reshape(s, (n_mels, -1, 1))
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
                'inputs': tf.TensorShape([n_mels, None, 1]),
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


def model(
    inputs,
    is_training = True,
    dropout_keep_prob = 0.8,
    reuse = None,
    scope = 'InceptionV4',
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
                    logits = slim.fully_connected(
                        bottleneck,
                        2,
                        activation_fn = None,
                        scope = 'Logits_vad',
                    )
                    return logits


init_lr = 1e-3
epochs = 300000
init_checkpoint = 'output-inception-v4/model.ckpt-401000'


def model_fn(features, labels, mode, params):
    Y = tf.cast(features['targets'][:, 0], tf.int32)

    with slim.arg_scope(inception_utils.inception_arg_scope()):
        logits = model(features['inputs'])

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

    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(
        variables, init_checkpoint
    )

    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

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
train_dataset = get_dataset(train_files, batch_size = 32)

dev_files = glob('vad2/data/vad-dev-*') + glob('noise/data/vad-dev-*')
dev_dataset = get_dataset(dev_files, batch_size = 16)

save_directory = 'output-inception-v4-vad'

train.run_training(
    train_fn = train_dataset,
    model_fn = model_fn,
    model_dir = save_directory,
    num_gpus = 1,
    log_step = 1,
    save_checkpoint_step = 25000,
    max_steps = epochs,
    eval_fn = dev_dataset,
    train_hooks = train_hooks,
)
