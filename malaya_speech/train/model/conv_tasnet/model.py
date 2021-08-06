# https://github.com/kaparoo/Conv-TasNet

import tensorflow as tf
from .layer import Encoder, Separater, Decoder


class ConvTasNet(tf.keras.Model):

    def __init__(self, param, **kwargs):
        super(ConvTasNet, self).__init__(**kwargs)
        self.param = param
        self.encoder = Encoder(param)
        self.separater = Separater(param)
        self.decoder = Decoder(param)

        self.concat = tf.keras.layers.concatenate
        self.reshape = tf.keras.layers.Reshape(
            target_shape=[param.That, param.C, param.N])
        self.permute = tf.keras.layers.Permute([2, 1, 3])
        self.apply_mask = tf.keras.layers.Multiply()

    def call(self, mixture_segments):
        # (, That, L) -> (, That, N)
        mixture_weights = self.encoder(mixture_segments)
        # (, That, N) -> (, C, That, N)
        source_masks = self.separater(mixture_weights)
        # (, That, N) -> (, C, That, N)
        mixture_weights = self.permute(self.reshape(self.concat(
            [mixture_weights for _ in range(self.param.C)], axis=-1)))
        # (, C, That, N), (, C, That, N) -> (, C, That, N)
        source_weights = self.apply_mask([mixture_weights, source_masks])
        # (, C, That, N) -> (, C, That, L)
        return self.decoder(source_weights)
