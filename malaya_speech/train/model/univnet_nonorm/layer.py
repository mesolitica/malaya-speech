import tensorflow as tf
import numpy as np
from ..utils import shape_list
from ..universal_melgan.model import TFConvTranspose1d


def get_initializer(initializer_seed=42):
    """Creates a `tf.initializers.glorot_normal` with the given seed.
    Args:
        initializer_seed: int, initializer seed.
    Returns:
        GlorotNormal initializer with seed = `initializer_seed`.
    """
    return tf.keras.initializers.glorot_normal(seed=initializer_seed)


"""
KernelPredictor(
    cond_channels=cond_channels, = 80,
    conv_in_channels=in_channels, = 32,
    conv_out_channels=2 * in_channels, = 64,
    conv_layers=conv_layers, = 4,
    conv_kernel_size=conv_kernel_size,
    kpnet_hidden_channels=kpnet_hidden_channels,
    kpnet_conv_size=kpnet_conv_size,
    kpnet_dropout=kpnet_dropout,
)
"""


class KernelPredictor(tf.keras.layers.Layer):
    def __init__(
        self,
        cond_channels,
        conv_in_channels,
        conv_out_channels,
        conv_layers,
        conv_kernel_size=3,
        kpnet_hidden_channels=64,
        kpnet_conv_size=3,
        kpnet_dropout=0.0,
        kpnet_nonlinear_activation='LeakyReLU',
        kpnet_nonlinear_activation_params={'alpha': 0.2},
        initializer_seed=42,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.conv_in_channels = conv_in_channels
        self.conv_out_channels = conv_out_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_layers = conv_layers

        l_w = conv_in_channels * conv_out_channels * conv_kernel_size * conv_layers
        l_b = conv_out_channels * conv_layers

        self.input_conv = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(
                filters=kpnet_hidden_channels,
                kernel_size=5,
                use_bias=True,
                padding='same',
                kernel_initializer=get_initializer(initializer_seed),
            ),
            getattr(tf.keras.layers, kpnet_nonlinear_activation)(
                **kpnet_nonlinear_activation_params
            )])

        padding = (kpnet_conv_size - 1) // 2
        self.residual_convs = []
        for _ in range(3):
            s = tf.keras.models.Sequential([
                tf.keras.layers.Dropout(kpnet_dropout),
                tf.keras.layers.Conv1D(kpnet_hidden_channels,
                                       kpnet_conv_size, padding='same', use_bias=True),
                getattr(tf.keras.layers, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
                tf.keras.layers.Conv1D(kpnet_hidden_channels,
                                       kpnet_conv_size, padding='same', use_bias=True),
                getattr(tf.keras.layers, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
            ])
            self.residual_convs.append(s)

        self.kernel_conv = tf.keras.layers.Conv1D(
            l_w, kpnet_conv_size, padding='same', use_bias=True)
        self.bias_conv = tf.keras.layers.Conv1D(
            l_b, kpnet_conv_size, padding='same', use_bias=True)

    def call(self, c, **kwargs):
        batch, cond_length, _ = shape_list(c)

        c = self.input_conv(c)
        for residual_conv in self.residual_convs:
            c = c + residual_conv(c)
        k = self.kernel_conv(c)
        b = self.bias_conv(c)
        kernels = tf.reshape(k, (batch, self.conv_layers, cond_length, self.conv_in_channels,
                                 self.conv_out_channels, self.conv_kernel_size))
        bias = tf.reshape(b, (batch, self.conv_layers, cond_length, self.conv_out_channels))
        return kernels, bias


"""
LVCBlock(
    channel_size = 32,
    hp.audio.n_mel_channels = 80,
    stride=stride, = [8,8,4],
    dilations=hp.gen.dilations, = [1, 3, 9, 27],
    cond_hop_length=hop_length, = hop_length = stride * hop_length
    kpnet_conv_size=kpnet_conv_size = 3
)
"""


class LVCBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        in_channels,
        cond_channels,
        stride,
        dilations=[1, 3, 9, 27],
        conv_kernel_size=3,
        cond_hop_length=256,
        kpnet_hidden_channels=64,
        kpnet_conv_size=3,
        kpnet_dropout=0.0,
        initializer_seed=42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cond_hop_length = cond_hop_length
        self.conv_layers = len(dilations)
        self.conv_kernel_size = conv_kernel_size

        self.kernel_predictor = KernelPredictor(
            cond_channels=cond_channels,
            conv_in_channels=in_channels,
            conv_out_channels=2 * in_channels,
            conv_layers=len(dilations),
            conv_kernel_size=conv_kernel_size,
            kpnet_hidden_channels=kpnet_hidden_channels,
            kpnet_conv_size=kpnet_conv_size,
            kpnet_dropout=kpnet_dropout,
            name='KernelPredictor'
        )

        self.convt_pre = tf.keras.models.Sequential([
            tf.keras.layers.LeakyReLU(alpha=0.2),
            TFConvTranspose1d(
                in_channels,
                kernel_size=stride * 2,
                strides=stride,
                padding='same',
                is_weight_norm=False,
                initializer_seed=initializer_seed,
            )])

        self.convs = []
        for d in dilations:
            conv = tf.keras.models.Sequential([
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Conv1D(
                    in_channels, kernel_size=conv_kernel_size, padding='same', dilation_rate=d
                ),
                tf.keras.layers.LeakyReLU(alpha=0.2),
            ])
            self.convs.append(conv)

    def call(self, x, c, **kwargs):
        """
        x (Tensor): the input sequence (batch, in_length, in_channels), the noise sequence (batch, in_length, noise_dim)
        c (Tensor): the conditioning sequence (batch, cond_channels, cond_length),  the conditioning sequence of mel-spectrogram (batch, in_length, mel_channels) 
        """
        _, _, in_channels = shape_list(x)
        x = self.convt_pre(x)
        kernels, bias = self.kernel_predictor(c)

        for i in range(self.conv_layers):
            output = self.convs[i](x)

            k = kernels[:, i, :, :, :, :]
            b = bias[:, i, :, :]

            output = self.location_variable_convolution(output, k, b, hop_size=self.cond_hop_length)
            x = x + tf.nn.sigmoid(output[:, :, :in_channels]) * tf.nn.tanh(output[:, :, in_channels:])

        return x

    def location_variable_convolution(self, x, kernel, bias, dilation=1, hop_size=256):
        """ 
        perform location-variable convolution operation on the input sequence (x) using the local convolution kernl. 
        Args:
            x (Tensor): the input sequence (batch, in_length, in_channels). 
            kernel (Tensor): the local convolution kernel (batch, kernel_length, in_channel, out_channels, kernel_size) 
            bias (Tensor): the bias for the local convolution (batch, kernel_length, out_channels) 
            dilation (int): the dilation of convolution. 
            hop_size (int): the hop_size of the conditioning sequence. 
        Returns:
            (Tensor): the output sequence after performing local convolution. (batch, in_length, out_channels).
        """
        batch, in_length, channel = shape_list(x)
        batch, kernel_length, _, out_channels, kernel_size = shape_list(kernel)

        padding = dilation * int((kernel_size - 1) / 2)
        output = tf.pad(x, ((0, 0), (padding, padding), (0, 0)))
        output = tf.image.extract_patches(tf.expand_dims(output, -1),
                                          sizes=[1, hop_size + 2 * padding, 1, 1],
                                          strides=[1, hop_size, 1, 1],
                                          rates=[1, 1, 1, 1],
                                          padding='VALID')
        output_shape = shape_list(output)

        padding = hop_size - dilation
        output = tf.pad(output, ((0, 0), (0, 0), (0, 0), (padding, padding)), mode='REFLECT')
        output = tf.reshape(output, (batch, output_shape[1], output_shape[2], -1, kernel_size))
        output = tf.expand_dims(output, -3)

        output = tf.einsum('blidsk,bliok->blsdo', output, kernel)
        output = output + tf.expand_dims(tf.expand_dims(bias, -2), -2)
        return tf.reshape(output, [batch, in_length, out_channels])
