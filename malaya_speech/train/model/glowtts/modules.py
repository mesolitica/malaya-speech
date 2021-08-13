import tensorflow as tf
import numpy as np
from ..utils import WeightNormalization


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = tf.tanh(in_act[:, :n_channels_int, :])
    s_act = tf.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class ConvReluNorm(tf.keras.layers.Layer):
    def __init__(self, config, name='ConvReluNorm', **kwargs):
        self.in_channels = config.in_channels
        self.hidden_channels = config.hidden_channels
        self.out_channels = config.out_channels
        self.kernel_size = config.kernel_size
        self.n_layers = config.n_layers
        self.p_dropout = config.p_dropout
        assert self.n_layers > 1, 'Number of layers should be larger than 0.'

        self.conv_layers = []
        self.norm_layers = []

        self.relu_drop = tf.keras.Sequential([tf.keras.layers.ReLU(),
                                              tf.keras.layers.Dropout(self.p_dropout)])

        for i in range(self.n_layers-1):
            self.conv_layers.append(
                tf.keras.layers.Conv1D(
                    self.hidden_channels,
                    self.kernel_size,
                    padding='same',
                    name='conv_._{}'.format(i),
                )
            )
            self.norm_layers.append(tf.keras.layers.LayerNormalization(
                epsilon=1e-4,
                name='LayerNorm_._{}'.format(i),
            ))

        self.proj = tf.keras.layers.Conv1D(self.out_channels, 1,
                                           kernel_initializer='zeros', bias_initializer='zeros',
                                           padding='same')

    def call(self, x, x_mask, training=False):
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask, training=training)
        x = self.norm_layers[i](x, training=training)
        x = self.relu_drop(x, training=training)
        x = x_org + self.proj(x, training=training)
        return x * x_mask


class WN(tf.keras.layers.Layer):
    def __init__(self, config, name='WN', **kwargs):
        assert(config.kernel_size % 2 == 1)
        assert(config.hidden_channels % 2 == 0)
        self.in_channels = config.in_channels
        self.hidden_channels = config.hidden_channels
        self.kernel_size = config.kernel_size,
        self.dilation_rate = config.dilation_rate
        self.n_layers = config.n_layers
        self.gin_channels = config.gin_channels
        self.p_dropout = config.p_dropout

        self.in_layers = []
        self.res_skip_layers = []
        self.drop = tf.keras.layers.Dropout(self.p_dropout)

        if self.gin_channels != 0:
            cond_layer = tf.keras.layers.Conv1D(2*self.hidden_channels*self.n_layers, 1, padding='same')
            self.cond_layer = WeightNormalization(cond_layer, data_init=False)

        for i in range(n_layers):
            dilation = dilation_rate ** i
            in_layer = tf.keras.layers.Conv1D(2*self.hidden_channels, self.kernel_size,
                                              dilation_rate=dilation, padding='same')
            in_layer = WeightNormalization(in_layer, data_init=False)
            self.in_layers.append(in_layer)

            if i < n_layers - 1:
                res_skip_channels = 2 * self.hidden_channels
            else:
                res_skip_channels = self.hidden_channels

            res_skip_layer = tf.keras.layers.Conv1D(res_skip_channels, 1)
            res_skip_layer = WeightNormalization(res_skip_layer, data_init=False)
            self.res_skip_layers.append(res_skip_layer)

    def call(self, x, x_mask=None, g=None, training=False):
        output = tf.zeros_like(x)
        n_channels_tensor = tf.constant([self.hidden_channels], dtype=tf.int32)

        if g is not None:
            g = self.cond_layer(g, training=training)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x, training=training)
            x_in = self.drop(x_in, training=training)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset+2*self.hidden_channels, :]
            else:
                g_l = tf.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(
                x_in,
                g_l,
                n_channels_tensor)

            res_skip_acts = self.res_skip_layers[i](acts, training=training)
            if i < self.n_layers - 1:
                x = (x + res_skip_acts[:, :self.hidden_channels, :]) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts

        return output * x_mask


class ActNorm(tf.keras.layers.Layer):
    def __init__(self, channels, ddi=False, **kwargs):
        self.channels = channels
        self.initialized = not ddi

        self.logs = tf.get_variable(
            name='logs',
            shape=[1, channels, 1],
            initializer=tf.zeros_initializer(),
        )
        self.bias = tf.get_variable(
            name='bias',
            shape=[1, channels, 1],
            initializer=tf.zeros_initializer(),
        )

    def forward(self, x, x_mask=None, reverse=False, **kwargs):
        x_len = tf.shape(x)[1]
        if x_mask is None:

            x_mask = torch.ones(x.size(0), 1, x.size(2)).to(device=x.device, dtype=x.dtype)
        x_len = torch.sum(x_mask, [1, 2])
