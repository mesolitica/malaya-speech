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

import tensorflow as tf  # pylint: disable=g-bad-import-order

from . import attention_layer
from . import ffn_layer
from . import model_utils

EOS_ID = 1
_NEG_INF = -1e9


class Transformer(object):
    """Transformer model for sequence to sequence data.

  Implemented as described in: https://arxiv.org/pdf/1706.03762.pdf

  The Transformer model consists of an encoder and decoder. The input is an int
  sequence (or a batch of sequences). The encoder produces a continous
  representation, and the decoder uses the encoder output to generate
  probabilities for the output sequence.
  """

    def __init__(self, params, train):
        self.train = train
        self.params = params
        self.encoder_stack = EncoderStack(params, train)
        self.decoder_stack = DecoderStack(params, train)

    def __call__(self, inputs, inputs_padding = None, targets = None):
        """Calculate target logits or inferred target sequences.

    Args:
      inputs: int tensor with shape [batch_size, input_length].
      targets: None or int tensor with shape [batch_size, target_length].

    Returns:
      If targets is defined, then return logits for each word in the target
      sequence. float tensor with shape [batch_size, target_length, vocab_size]
      If target is none, then generate output sequence one token at a time.
        returns a dictionary {
          output: [batch_size, decoded length]
          score: [batch_size, float]}
    """
        # Variance scaling is used here because it seems to work in many problems.
        # Other reasonable initializers may also work just as well.
        initializer = tf.variance_scaling_initializer(
            self.params['initializer_gain'],
            mode = 'fan_avg',
            distribution = 'uniform',
        )
        if inputs_padding is None:
            inputs_padding = tf.fill(
                (tf.shape(inputs)[0], tf.shape(inputs)[1]), 0.0
            )
        with tf.variable_scope('Transformer', initializer = initializer):
            attention_bias = model_utils.get_padding_bias(inputs)
            encoder_outputs = self.encode(
                inputs, inputs_padding, attention_bias
            )
            if targets is None:
                targets = encoder_outputs
            logits = self.decode(targets, encoder_outputs, attention_bias)
            return logits

    def encode(self, inputs, inputs_padding, attention_bias):
        with tf.name_scope('encode'):

            with tf.name_scope('add_pos_encoding'):
                length = tf.shape(embedded_inputs)[1]
                pos_encoding = model_utils.get_position_encoding(
                    length, self.params['hidden_size']
                )
                encoder_inputs = embedded_inputs + pos_encoding

            if self.train:
                encoder_inputs = tf.nn.dropout(
                    encoder_inputs, 1 - self.params['layer_postprocess_dropout']
                )

            return self.encoder_stack(
                encoder_inputs, attention_bias, inputs_padding
            )

    def decode(self, targets, encoder_outputs, attention_bias):
        with tf.name_scope('decode'):
            decoder_inputs = targets
            with tf.name_scope('add_pos_encoding'):
                length = tf.shape(decoder_inputs)[1]
                decoder_inputs += model_utils.get_position_encoding(
                    length, self.params['hidden_size']
                )
            if self.train:
                decoder_inputs = tf.nn.dropout(
                    decoder_inputs, 1 - self.params['layer_postprocess_dropout']
                )
            decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
                length
            )
            outputs = self.decoder_stack(
                decoder_inputs,
                encoder_outputs,
                decoder_self_attention_bias,
                attention_bias,
            )
            return outputs


class LayerNormalization(tf.layers.Layer):
    """Applies layer normalization."""

    def __init__(self, hidden_size):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size

    def build(self, _):
        self.scale = tf.get_variable(
            'layer_norm_scale',
            [self.hidden_size],
            initializer = tf.ones_initializer(),
        )
        self.bias = tf.get_variable(
            'layer_norm_bias',
            [self.hidden_size],
            initializer = tf.zeros_initializer(),
        )
        self.built = True

    def call(self, x, epsilon = 1e-6):
        mean = tf.reduce_mean(x, axis = [-1], keepdims = True)
        variance = tf.reduce_mean(
            tf.square(x - mean), axis = [-1], keepdims = True
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
    """Transformer encoder stack.

  The encoder stack is made up of N identical layers. Each layer is composed
  of the sublayers:
    1. Self-attention layer
    2. Feedforward network (which is 2 fully-connected layers)
  """

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
                activation = params.get('activation', 'relu'),
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
        """Return the output of the encoder layer stacks.

    Args:
      encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
      attention_bias: bias for the encoder self-attention layer.
        [batch_size, 1, 1, input_length]
      inputs_padding: P

    Returns:
      Output of encoder layer stack.
      float32 tensor with shape [batch_size, input_length, hidden_size]
    """
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


class DecoderStack(tf.layers.Layer):
    """Transformer decoder stack.

  Like the encoder stack, the decoder stack is made up of N identical layers.
  Each layer is composed of the sublayers:
    1. Self-attention layer
    2. Multi-headed attention layer combining encoder outputs with results from
       the previous self-attention layer.
    3. Feedforward network (2 fully-connected layers)
  """

    def __init__(self, params, train):
        super(DecoderStack, self).__init__()
        self.layers = []
        for _ in range(params['num_hidden_layers']):
            self_attention_layer = attention_layer.SelfAttention(
                params['hidden_size'],
                params['num_heads'],
                params['attention_dropout'],
                train,
            )
            enc_dec_attention_layer = attention_layer.Attention(
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
                activation = params.get('activation', 'relu'),
            )

            self.layers.append(
                [
                    PrePostProcessingWrapper(
                        self_attention_layer, params, train
                    ),
                    PrePostProcessingWrapper(
                        enc_dec_attention_layer, params, train
                    ),
                    PrePostProcessingWrapper(
                        feed_forward_network, params, train
                    ),
                ]
            )

        self.output_normalization = LayerNormalization(params['hidden_size'])

    def call(
        self,
        decoder_inputs,
        encoder_outputs,
        decoder_self_attention_bias,
        attention_bias,
        cache = None,
    ):
        """Return the output of the decoder layer stacks.

    Args:
      decoder_inputs: tensor with shape [batch_size, target_length, hidden_size]
      encoder_outputs: tensor with shape [batch_size, input_length, hidden_size]
      decoder_self_attention_bias: bias for decoder self-attention layer.
        [1, 1, target_len, target_length]
      attention_bias: bias for encoder-decoder attention layer.
        [batch_size, 1, 1, input_length]
      cache: (Used for fast decoding) A nested dictionary storing previous
        decoder self-attention values. The items are:
          {layer_n: {"k": tensor with shape [batch_size, i, key_channels],
                     "v": tensor with shape [batch_size, i, value_channels]},
           ...}

    Returns:
      Output of decoder layer stack.
      float32 tensor with shape [batch_size, target_length, hidden_size]
    """
        for n, layer in enumerate(self.layers):
            self_attention_layer = layer[0]
            enc_dec_attention_layer = layer[1]
            feed_forward_network = layer[2]

            # Run inputs through the sublayers.
            layer_name = 'layer_%d' % n
            layer_cache = cache[layer_name] if cache is not None else None
            with tf.variable_scope(layer_name):
                with tf.variable_scope('self_attention'):
                    decoder_inputs = self_attention_layer(
                        decoder_inputs,
                        decoder_self_attention_bias,
                        cache = layer_cache,
                    )
                with tf.variable_scope('encdec_attention'):
                    decoder_inputs = enc_dec_attention_layer(
                        decoder_inputs, encoder_outputs, attention_bias
                    )
                with tf.variable_scope('ffn'):
                    decoder_inputs = feed_forward_network(decoder_inputs)

        return self.output_normalization(decoder_inputs)
