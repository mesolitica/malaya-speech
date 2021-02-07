import tensorflow as tf
from ..conformer.model import Model as ConformerModel
from ..fastspeech.model import TFTacotronPostnet
import numpy as np
import copy


class Encoder(tf.keras.layers.Layer):
    def __init__(self, dim_neck, config, **kwargs):
        super(Encoder, self).__init__(name = 'Encoder', **kwargs)
        self.config = config
        self.encoder = ConformerModel(**self.config)
        self.encoder_dense = tf.keras.layers.Dense(
            units = dim_neck, dtype = tf.float32, name = 'encoder_dense'
        )

    def call(self, x, c_org, training = True):
        c_org = tf.tile(tf.expand_dims(c_org, 1), (1, tf.shape(x)[1], 1))
        x = tf.concat([x, c_org], axis = -1)
        f = self.encoder(x)
        return self.encoder_dense(f)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(Decoder, self).__init__(name = 'Decoder', **kwargs)
        self.config = config
        self.encoder = ConformerModel(**self.config)

    def call(self, x, training = True):
        return self.encoder(x)


class Model(tf.keras.Model):
    def __init__(self, dim_neck, config, config_fastspeech, num_mels, **kwargs):
        super(Model, self).__init__(name = 'conformervc', **kwargs)
        self.encoder = Encoder(dim_neck, copy.deepcopy(config))
        self.decoder = Decoder(copy.deepcopy(config))
        self.mel_dense = tf.keras.layers.Dense(
            units = num_mels, dtype = tf.float32, name = 'mel_before'
        )
        self.postnet = TFTacotronPostnet(
            config = config_fastspeech, dtype = tf.float32, name = 'postnet'
        )
        self.config = config

    def call_second(self, x, c_org, mel_lengths, training = True):
        max_length = tf.cast(tf.reduce_max(mel_lengths), tf.int32)
        attention_mask = tf.sequence_mask(
            lengths = mel_lengths, maxlen = max_length, dtype = tf.float32
        )
        attention_mask.set_shape((None, None))
        extended_mask = tf.cast(
            tf.expand_dims(attention_mask, axis = 2), x.dtype
        )
        return self.encoder(x, c_org, training = training) * extended_mask

    def call(self, x, c_org, c_trg, mel_lengths, training = True, **kwargs):

        max_length = tf.cast(tf.reduce_max(mel_lengths), tf.int32)
        attention_mask = tf.sequence_mask(
            lengths = mel_lengths, maxlen = max_length, dtype = tf.float32
        )
        attention_mask.set_shape((None, None))
        extended_mask = tf.cast(
            tf.expand_dims(attention_mask, axis = 2), x.dtype
        )
        code_exp = self.encoder(x, c_org, training = training) * extended_mask
        c_trg = tf.tile(tf.expand_dims(c_trg, 1), (1, tf.shape(x)[1], 1))
        encoder_outputs = tf.concat([code_exp, c_trg], axis = -1)
        decoder_output = (
            self.decoder(encoder_outputs, training = training) * extended_mask
        )
        mel_before = self.mel_dense(decoder_output)
        mel_after = (
            self.postnet([mel_before, attention_mask], training = training)
            + mel_before
        )

        return encoder_outputs, mel_before, mel_after, code_exp
