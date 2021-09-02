import tensorflow as tf
import numpy as np
from . import modules
from ..utils import WeightNormalization, shape_list


class CouplingBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0, sigmoid_scale=False,
                 name=0, **kwargs):
        super(CouplingBlock, self).__init__(name=f'CouplingBlock_{name}', **kwargs)
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        self.sigmoid_scale = sigmoid_scale

        start = tf.keras.layers.Conv1D(self.hidden_channels, 1, padding='same')
        self.start = WeightNormalization(start, data_init=False)

        self.end = tf.keras.layers.Conv1D(self.in_channels, 1,
                                          kernel_initializer='zeros', bias_initializer='zeros',
                                          padding='same')

        self.wn = modules.WN(in_channels, hidden_channels, kernel_size,
                             dilation_rate, n_layers, gin_channels, p_dropout)

    def call(self, x, x_mask=None, reverse=False, g=None, training=True):
        b, t, c = shape_list(x)
        if x_mask is None:
            x_mask = 1.0
        x_0, x_1 = x[:, :, :self.in_channels//2], x[:, :, self.in_channels//2:]
        x = self.start(x_0, training=training) * x_mask
        x = self.wn(x, x_mask, g, training=training)
        out = self.end(x, training=training)

        z_0 = x_0
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

        z = tf.concat([z_0, z_1], 2)
        return z, logdet

    def store_inverse(self):
        self.start = self.start.remove()
