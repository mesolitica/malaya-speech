# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

SUPPORTED_RNNS = {
    'lstm': tf.keras.layers.LSTMCell,
    'rnn': tf.keras.layers.SimpleRNNCell,
    'gru': tf.keras.layers.GRUCell,
}

# Parameters for batch normalization.
_BATCH_NORM_EPSILON = 1e-5
_BATCH_NORM_DECAY = 0.997

# Filters of convolution layer
_CONV_FILTERS = 32


def batch_norm(inputs, training):
    return tf.keras.layers.BatchNormalization(
        momentum = _BATCH_NORM_DECAY, epsilon = _BATCH_NORM_EPSILON
    )(inputs, training = training)


def _conv_bn_layer(
    inputs, padding, filters, kernel_size, strides, layer_id, training
):
    inputs = tf.pad(
        inputs,
        [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]],
    )
    inputs = tf.keras.layers.Conv2D(
        filters = filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = 'valid',
        use_bias = False,
        activation = tf.nn.relu6,
        name = 'cnn_{}'.format(layer_id),
    )(inputs)
    return batch_norm(inputs, training)


def _rnn_layer(
    inputs,
    rnn_cell,
    rnn_hidden_size,
    layer_id,
    is_batch_norm,
    is_bidirectional,
    training,
):
    if is_batch_norm:
        inputs = batch_norm(inputs, training)

    if is_bidirectional:
        rnn_outputs = tf.keras.layers.Bidirectional(
            tf.keras.layers.RNN(
                rnn_cell(rnn_hidden_size), return_sequences = True
            )
        )(inputs)
    else:
        rnn_outputs = tf.keras.layers.RNN(
            rnn_cell(rnn_hidden_size), return_sequences = True
        )(inputs)

    return rnn_outputs


class Model:
    def __init__(
        self,
        inputs,
        num_rnn_layers = 5,
        rnn_type = 'gru',
        is_bidirectional = True,
        rnn_hidden_size = 512,
        use_bias = True,
        training = True,
        **kwargs
    ):
        inputs = _conv_bn_layer(
            inputs,
            padding = (20, 5),
            filters = _CONV_FILTERS,
            kernel_size = (41, 11),
            strides = (2, 2),
            layer_id = 1,
            training = training,
        )
        inputs = _conv_bn_layer(
            inputs,
            padding = (10, 5),
            filters = _CONV_FILTERS,
            kernel_size = (21, 11),
            strides = (2, 1),
            layer_id = 2,
            training = training,
        )
        batch_size = tf.shape(inputs)[0]
        feat_size = inputs.get_shape().as_list()[2]
        inputs = tf.reshape(inputs, [batch_size, -1, feat_size * _CONV_FILTERS])
        rnn_cell = SUPPORTED_RNNS[rnn_type]
        for layer_counter in range(num_rnn_layers):
            is_batch_norm = layer_counter != 0
            inputs = _rnn_layer(
                inputs,
                rnn_cell,
                rnn_hidden_size,
                layer_counter + 1,
                is_batch_norm,
                is_bidirectional,
                training,
            )
        self.logits = batch_norm(inputs, training)
