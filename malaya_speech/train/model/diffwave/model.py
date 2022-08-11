import tensorflow as tf
import math
import numpy as np
from ..initializer import HeNormal
from ..utils import shape_list
from ..melgan.layer import WeightNormalization


def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)

    _embed = tf.exp(tf.range(half_dim))
    _embed = diffusion_steps * _embed
    diffusion_step_embed = tf.concat([tf.math.sin(_embed), tf.math.cos(_embed)], axis=-1)
    return diffusion_step_embed


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, res_channels, skip_channels, dilation,
                 diffusion_step_embed_dim_out, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)

        initializer = HeNormal()

        self.res_channels = res_channels

        # Use a FC layer for diffusion step embedding
        self.fc_t = tf.keras.layers.Dense(self.res_channels)

        self.dilated_conv_layer = tf.keras.layers.Conv1D(2 * self.res_channels, kernel_size=3, padding='SAME',
                                                         dilation_rate=dilation, kernel_initializer=initializer)

        self.upsample_conv2d = []
        for s in [16, 16]:

            conv_trans2d = tf.keras.layers.Conv2DTranspose(
                1,
                (3, 2 * s),
                strides=(1, s),
                padding='SAME',
                kernel_initializer=initializer)
            self.upsample_conv2d.append(conv_trans2d)

        self.mel_conv = tf.keras.layers.Conv1D(2 * self.res_channels, kernel_size=1, padding 'SAME',
                                               kernel_initializer=initializer)

        self.res_conv = WeightNormalization(tf.keras.layers.Conv1D(res_channels, kernel_size=1, padding 'SAME',
                                                                   kernel_initializer=initializer))

        self.skip_conv = WeightNormalization(tf.keras.layers.Conv1D(skip_channels, kernel_size=1, padding 'SAME',
                                                                    kernel_initializer=initializer))

    def call(self, input_data):
        x, mel_spec, diffusion_step_embed = input_data
        h = x
        batch_size, seq_len, n_channels = shape_list(x)

        part_t = self.fc_t(diffusion_step_embed)
        part_t = tf.reshape(part_t, [batch_size, 1, self.res_channels])
        h += part_t

        h = self.dilated_conv_layer(h)
        mel_spec = torch.unsqueeze(mel_spec, dim=1)
