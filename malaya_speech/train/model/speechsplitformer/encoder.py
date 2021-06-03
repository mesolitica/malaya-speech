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
# ==============================================================================
"""Defines the Transformer model, and its encoder and decoder stacks.

Model paper: https://arxiv.org/pdf/1706.03762.pdf
Transformer model code source: https://github.com/tensorflow/tensor2tensor
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ..transformer import attention_layer
from ..transformer import ffn_layer
from ..transformer import model_utils
from ..speechsplit.model import InterpLnr

EOS_ID = 1
_NEG_INF = -1e9


class LayerNormalization(tf.layers.Layer):
    """Applies layer normalization."""

    def __init__(self, hidden_size):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size

    def build(self, _):
        self.scale = tf.get_variable(
            'layer_norm_scale',
            [self.hidden_size],
            initializer=tf.ones_initializer(),
        )
        self.bias = tf.get_variable(
            'layer_norm_bias',
            [self.hidden_size],
            initializer=tf.zeros_initializer(),
        )
        self.built = True

    def call(self, x, epsilon=1e-6):
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(
            tf.square(x - mean), axis=[-1], keepdims=True
        )
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * self.scale + self.bias


class PrePostProcessingWrapper(object):
    """Wrapper class that applies layer pre-processing and post-processing."""

    def __init__(self, layer, params, train):
        self.layer = layer
        self.postprocess_dropout = params['layer_postprocess_dropout']
        self.train = train

        # Create normalization layer
        self.layer_norm = LayerNormalization(params['hidden_size'])

    def __call__(self, x, *args, **kwargs):
        # Preprocessing: apply layer normalization
        y = self.layer_norm(x)

        # Get layer output
        y = self.layer(y, *args, **kwargs)

        # Postprocessing: apply dropout and residual connection
        if self.train:
            y = tf.nn.dropout(y, 1 - self.postprocess_dropout)
        return x + y


class EncoderStack(tf.layers.Layer):
    def __init__(self, params, train):
        super(EncoderStack, self).__init__()
        self.layers = []

        for _ in range(params['num_hidden_layers']):
            # Create sublayers for each layer.
            self_attention_layer = attention_layer.SelfAttention(
                params['hidden_size'],
                params['num_heads'],
                params['attention_dropout'],
                train,
            )
            feed_forward_network = ffn_layer.FeedFowardNetwork(
                params['hidden_size'],
                params['filter_size'],
                params['relu_dropout'],
                train,
                params['allow_ffn_pad'],
                activation=params.get('activation', 'relu'),
            )

            self.layers.append(
                [
                    PrePostProcessingWrapper(
                        self_attention_layer, params, train
                    ),
                    PrePostProcessingWrapper(
                        feed_forward_network, params, train
                    ),
                ]
            )

        self.output_normalization = LayerNormalization(params['hidden_size'])

    def call(self, encoder_inputs, attention_bias, inputs_padding):
        for n, layer in enumerate(self.layers):
            # Run inputs through the sublayers.
            self_attention_layer = layer[0]
            feed_forward_network = layer[1]

            with tf.variable_scope('layer_%d' % n):
                with tf.variable_scope('self_attention'):
                    encoder_inputs = self_attention_layer(
                        encoder_inputs, attention_bias
                    )
                with tf.variable_scope('ffn'):
                    encoder_inputs = feed_forward_network(
                        encoder_inputs, inputs_padding
                    )

        return self.output_normalization(encoder_inputs)
