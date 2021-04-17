import tensorflow as tf
from ..fastspeech.model import TFTacotronPostnet
from ..transformer.transformer import EncoderStack, DecoderStack
from ..transformer.model_utils import (
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
    def __init__(self, dim_neck, params, train, **kwargs):
        super(Encoder, self).__init__(name = 'Encoder', **kwargs)
        self.params = params
        self.encoder_stack = EncoderStack(params, train)
        self.encoder_dense = tf.keras.layers.Dense(
            units = dim_neck, dtype = tf.float32, name = 'encoder_dense'
        )

    def call(self, x, c_org, attention_mask, training = True):
        c_org = tf.tile(tf.expand_dims(c_org, 1), (1, tf.shape(x)[1], 1))
        x = tf.concat([x, c_org], axis = -1)
        initializer = tf.variance_scaling_initializer(
            self.params['initializer_gain'],
            mode = 'fan_avg',
            distribution = 'uniform',
        )
        inputs_padding = tf.cast(
            tf.logical_not(tf.cast(attention_mask, tf.bool)), tf.float32
        )
        with tf.variable_scope('Transformer', initializer = initializer):
            attention_bias = get_padding_bias(inputs_padding)

            with tf.name_scope('encode'):
                with tf.name_scope('add_pos_encoding'):
                    length = tf.shape(x)[1]
                    encoder_inputs = x

                if training:
                    encoder_inputs = tf.nn.dropout(
                        encoder_inputs,
                        1 - self.params['layer_postprocess_dropout'],
                    )

                return self.encoder_dense(
                    self.encoder_stack(
                        encoder_inputs, attention_bias, inputs_padding
                    )
                )


class Decoder(tf.keras.layers.Layer):
    def __init__(self, params, train, **kwargs):
        super(Decoder, self).__init__(name = 'Decoder', **kwargs)
        self.decoder_stack = DecoderStack(params, train)
        self.params = params

    def call(self, x, attention_mask, encoder_outputs, training = True):
        input_shape = tf.shape(x)
        seq_length = input_shape[1]

        initializer = tf.variance_scaling_initializer(
            self.params['initializer_gain'],
            mode = 'fan_avg',
            distribution = 'uniform',
        )

        inputs_padding = tf.cast(
            tf.logical_not(tf.cast(attention_mask, tf.bool)), tf.float32
        )

        with tf.variable_scope('Transformer', initializer = initializer):
            attention_bias = get_padding_bias(inputs_padding)

            with tf.name_scope('decode'):
                decoder_inputs = x
                with tf.name_scope('add_pos_encoding'):
                    length = tf.shape(decoder_inputs)[1]
                if training:
                    decoder_inputs = tf.nn.dropout(
                        decoder_inputs,
                        1 - self.params['layer_postprocess_dropout'],
                    )
                decoder_self_attention_bias = get_decoder_self_attention_bias(
                    length
                )
                return self.decoder_stack(
                    decoder_inputs,
                    encoder_outputs,
                    decoder_self_attention_bias,
                    attention_bias,
                )


class Model(tf.keras.Model):
    def __init__(
        self,
        dim_neck,
        params_encoder,
        params_decoder,
        config,
        dim_speaker = 0,
        train = True,
        **kwargs,
    ):
        super(Model, self).__init__(name = 'fastvc', **kwargs)
        self.encoder = Encoder(dim_neck, params_encoder, train = train)
        self.decoder = Decoder(params_decoder, train = train)
        self.mel_dense = tf.keras.layers.Dense(
            units = config.num_mels, dtype = tf.float32, name = 'mel_before'
        )
        self.postnet = TFTacotronPostnet(
            config = config, dtype = tf.float32, name = 'postnet'
        )
        if dim_speaker > 0:
            self.dim_speaker = tf.keras.layers.Dense(
                units = dim_speaker, dtype = tf.float32, name = 'dim_speaker'
            )
        else:
            self.dim_speaker = None

    def call_second(self, x, c_org, mel_lengths, training = True):
        max_length = tf.cast(tf.reduce_max(mel_lengths), tf.int32)
        attention_mask = tf.sequence_mask(
            lengths = mel_lengths, maxlen = max_length, dtype = tf.float32
        )
        attention_mask.set_shape((None, None))
        if self.dim_speaker:
            c_org = self.dim_speaker(c_org)
        code_exp = self.encoder(x, c_org, attention_mask, training = training)
        return code_exp

    def call(self, x, c_org, c_trg, mel_lengths, training = True, **kwargs):

        if self.dim_speaker:
            c_org = self.dim_speaker(c_org)
            c_trg = self.dim_speaker(c_trg)

        max_length = tf.cast(tf.reduce_max(mel_lengths), tf.int32)
        attention_mask = tf.sequence_mask(
            lengths = mel_lengths, maxlen = max_length, dtype = tf.float32
        )
        attention_mask.set_shape((None, None))
        code_exp = self.encoder(x, c_org, attention_mask, training = training)
        c_trg = tf.tile(tf.expand_dims(c_trg, 1), (1, tf.shape(x)[1], 1))
        encoder_outputs = tf.concat([code_exp, c_trg], axis = -1)
        decoder_output = self.decoder(
            encoder_outputs, attention_mask, code_exp, training = training
        )
        mel_before = self.mel_dense(decoder_output)
        mel_after = (
            self.postnet([mel_before, attention_mask], training = training)
            + mel_before
        )

        return encoder_outputs, mel_before, mel_after, code_exp
