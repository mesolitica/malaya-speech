import tensorflow as tf
from .transformer import EncoderStack, DecoderStack
from .model_utils import (
    get_position_encoding,
    get_padding_bias,
    get_decoder_self_attention_bias,
)
import numpy as np

_NEG_INF = -1e9


def get_padding_bias(padding):
    with tf.name_scope('attention_bias'):

        attention_bias = padding * _NEG_INF
        attention_bias = tf.expand_dims(
            tf.expand_dims(attention_bias, axis = 1), axis = 1
        )
    return attention_bias


class Encoder(tf.keras.layers.Layer):
    def __init__(self, params, train, **kwargs):
        super(Encoder, self).__init__(name = 'Encoder', **kwargs)
        self.params = params
        self.encoder_stack = EncoderStack(params, train)

    def call(self, x, attention_mask = None, training = True):
        initializer = tf.variance_scaling_initializer(
            self.params['initializer_gain'],
            mode = 'fan_avg',
            distribution = 'uniform',
        )
        if attention_mask is None:
            lengths = tf.tile([tf.shape(x)[1]], [tf.shape(x)[0]])
            max_length = tf.cast(tf.reduce_max(lengths), tf.int32)
            attention_mask = tf.sequence_mask(
                lengths = lengths, maxlen = max_length, dtype = tf.float32
            )
            attention_mask.set_shape((None, None))

        inputs_padding = tf.cast(
            tf.logical_not(tf.cast(attention_mask, tf.bool)), tf.float32
        )
        with tf.variable_scope('Transformer', initializer = initializer):
            attention_bias = get_padding_bias(inputs_padding)

            with tf.name_scope('encode'):
                with tf.name_scope('add_pos_encoding'):
                    length = tf.shape(x)[1]
                    pos_encoding = get_position_encoding(
                        length, self.params['hidden_size']
                    )
                    encoder_inputs = x + pos_encoding

                if training:
                    encoder_inputs = tf.nn.dropout(
                        encoder_inputs,
                        1 - self.params['layer_postprocess_dropout'],
                    )

                return self.encoder_stack(
                    encoder_inputs, attention_bias, inputs_padding
                )
