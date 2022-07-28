from ..universal_melgan.model import TFReflectionPad1d
from ..melgan.layer import WeightNormalization
from ..utils import shape_list
from .lvcnet import LVCBlock
import tensorflow as tf

MAX_WAV_VALUE = 32768.0


class Generator(tf.keras.Model):

    def __init__(self, hp, **kwargs):
        super().__init__(**kwargs)
        self.mel_channel = hp.audio.n_mel_channels
        self.noise_dim = hp.gen.noise_dim
        self.hop_length = hp.audio.hop_length
        channel_size = hp.gen.channel_size
        kpnet_conv_size = hp.gen.kpnet_conv_size

        kernel_size = 7

        self.conv_pre = tf.keras.models.Sequential([
            TFReflectionPad1d(
                (kernel_size - 1) // 2,
                padding_type='REFLECT',
            ),
            WeightNormalization(tf.keras.layers.Conv1D(
                filters=channel_size,
                kernel_size=kernel_size,
            ))
        ])

        self.conv_post = tf.keras.models.Sequential([
            tf.keras.layers.LeakyReLU(alpha=hp.gen.lReLU_slope),
            TFReflectionPad1d(
                (kernel_size - 1) // 2,
                padding_type='REFLECT',
            ),
            WeightNormalization(tf.keras.layers.Conv1D(1, kernel_size=kernel_size)),
            tf.keras.layers.Activation('tanh'),
        ])

        self.res_stack = []
        hop_length = 1
        for stride in hp.gen.strides:
            hop_length = stride * hop_length
            self.res_stack.append(
                LVCBlock(
                    channel_size,
                    hp.audio.n_mel_channels,
                    stride=stride,
                    dilations=hp.gen.dilations,
                    lReLU_slope=hp.gen.lReLU_slope,
                    cond_hop_length=hop_length,
                    kpnet_conv_size=kpnet_conv_size
                )
            )

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

    def inference(self, c, z=None):
        zero = tf.fill((1, 10, self.mel_channel), -11.5129)
        mel = tf.concat([c, zero], axis=1)

        if z is None:
            b, l, _ = shape_list(mel)
            z = torch.randn(1, l, self.noise_dim)

        audio = self.call(mel, z)
        return audio
