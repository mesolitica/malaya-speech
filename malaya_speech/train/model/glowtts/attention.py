import tensorflow as tf
import numpy as np
from . import modules
from ..utils import WeightNormalization, shape_list


class CouplingBlock(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(CouplingBlock, self).__init__(name='CouplingBlock', **kwargs)
        self.in_channels = config.in_channels
        self.hidden_channels = config.hidden_channels
        self.kernel_size = config.kernel_size
        self.dilation_rate = config.dilation_rate
        self.n_layers = config.n_layers
        self.gin_channels = config.gin_channels
        self.p_dropout = config.p_dropout
        self.sigmoid_scale = config.sigmoid_scale

        start = tf.keras.layers.Conv1D(self.hidden_channels, 1, padding='same')
        self.start = WeightNormalization(start, data_init=False)

        self.end = tf.keras.layers.Conv1D(self.hidden_channels, 1,
                                          kernel_initializer='zeros', bias_initializer='zeros',
                                          padding='same')

        self.wn = modules.WN(config)

    def forward(self, x, x_mask=None, reverse=False, g=None, training=True):
        b, t, c = shape_list(x)
        if x_mask is None:
            x_mask = 1.0
        x_0, x_1 = x[:, :, :self.in_channels//2], x[:, :, self.in_channels//2:]
        x = self.start(x_0, training=training) * x_mask
        x = self.wn(x, x_mask, g, training=training)
        out = self.end(x, training=training)

        _0 = x_0
        m = out[:, :, :self.in_channels//2]
        logs = out[:, :, self.in_channels//2:]
        if self.sigmoid_scale:
            logs = tf.math.log(1e-6 + tf.nn.sigmoid(logs + 2))

        if reverse:
            z_1 = (x_1 - m) * tf.math.exp(-logs) * x_mask
            logdet = None
        else:
            z_1 = (m + tf.math.exp(logs) * x_1) * x_mask
            logdet = tf.reduce_sum(logs * x_mask, [2, 1])

        z = tf.concat([z_0, z_1], 1)
        return z, logdet
