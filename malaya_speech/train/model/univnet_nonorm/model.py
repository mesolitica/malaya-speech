from ..universal_melgan.model import TFReflectionPad1d
from ..utils import shape_list
from .layer import get_initializer, LVCBlock
import tensorflow as tf


class Generator(tf.keras.Model):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.mel_channel = config.n_mel_channels
        self.noise_dim = config.noise_dim
        self.hop_length = config.hop_length
        channel_size = config.channel_size
        kpnet_conv_size = config.kpnet_conv_size
        kpnet_hidden_channels = config.kpnet_hidden_channels

        self.conv_pre = tf.keras.models.Sequential([
            TFReflectionPad1d(3),
            tf.keras.layers.Conv1D(
                filters=channel_size,
                kernel_size=7,
                dilation_rate=1,
                use_bias=True,
                kernel_initializer=get_initializer(config.initializer_seed),
            )
        ])

        self.res_stack = []
        hop_length = 1
        for stride in config.strides:
            hop_length = stride * hop_length
            self.res_stack.append(
                LVCBlock(
                    channel_size,
                    self.mel_channel,
                    stride=stride,
                    dilations=config.dilations,
                    cond_hop_length=hop_length,
                    kpnet_hidden_channels=kpnet_hidden_channels,
                    kpnet_conv_size=kpnet_conv_size
                )
            )

        self.conv_post = tf.keras.models.Sequential([
            tf.keras.layers.LeakyReLU(alpha=0.2),
            TFReflectionPad1d(3),
            tf.keras.layers.Conv1D(
                filters=1,
                kernel_size=7,
                dilation_rate=1,
                use_bias=True,
                kernel_initializer=get_initializer(config.initializer_seed),
            ),
            tf.keras.layers.Activation('tanh'),
        ])

    def call(self, c, z=None):
        """
        c (Tensor): the conditioning sequence of mel-spectrogram (batch, in_length, mel_channels) 
        z (Tensor): the noise sequence (batch, in_length, noise_dim)
        """
        if z is None:
            b, l, _ = shape_list(c)
            z = tf.random.normal(shape=(b, l, self.noise_dim))
        z = self.conv_pre(z)
        for res_block in self.res_stack:
            z = res_block(z, c)
        z = self.conv_post(z)
        return z
