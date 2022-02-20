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
        kpnet_nonlinear_activation_params={'alpha': 0.1},
        initializer_seed=0.02,
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

        self.residual_conv = tf.keras.models.Sequential([
            tf.keras.layers.Dropout(kpnet_dropout),
            tf.keras.layers.Conv1D(kpnet_hidden_channels, kpnet_conv_size, padding='same', bias=True),
            getattr(tf.keras.layers, kpnet_nonlinear_activation)(
                **kpnet_nonlinear_activation_params
            ),
            tf.keras.layers.Conv1D(kpnet_hidden_channels, kpnet_conv_size, padding='same', bias=True),
            getattr(tf.keras.layers, kpnet_nonlinear_activation)(
                **kpnet_nonlinear_activation_params
            ),
            tf.keras.layers.Dropout(kpnet_dropout),
            tf.keras.layers.Conv1D(kpnet_hidden_channels, kpnet_conv_size, padding='same', bias=True),
            getattr(tf.keras.layers, kpnet_nonlinear_activation)(
                **kpnet_nonlinear_activation_params
            ),
            tf.keras.layers.Conv1D(kpnet_hidden_channels, kpnet_conv_size, padding='same', bias=True),
            getattr(tf.keras.layers, kpnet_nonlinear_activation)(
                **kpnet_nonlinear_activation_params
            ),
            tf.keras.layers.Dropout(kpnet_dropout),
            tf.keras.layers.Conv1D(kpnet_hidden_channels, kpnet_conv_size, padding='same', bias=True),
            getattr(tf.keras.layers, kpnet_nonlinear_activation)(
                **kpnet_nonlinear_activation_params
            ),
            tf.keras.layers.Conv1D(kpnet_hidden_channels, kpnet_conv_size, padding='same', bias=True),
            getattr(tf.keras.layers, kpnet_nonlinear_activation)(
                **kpnet_nonlinear_activation_params
            ),
        ])

        self.kernel_conv = tf.keras.layers.Conv1D(l_w, kpnet_conv_size, padding='same', bias=True)
        self.bias_conv = tf.keras.layers.Conv1D(l_b, kpnet_conv_size, padding='same', bias=True)

    def call(self, c, **kwargs):
        batch, cond_length, _ = shape_list(c)

        c = self.input_conv(c)
        c = c + self.residual_conv(c)
        k = self.kernel_conv(c)
        b = self.bias_conv(c)
        kernels = tf.reshape(k, (batch, elf.conv_layers, self.conv_in_channels,
                             self.conv_out_channels, self.conv_kernel_size, cond_length))
        bias = tf.reshape(b, (batch, self.conv_layers, self.conv_out_channels, cond_length))
        return kernels, bias


class LVCBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        in_channels,
        cond_channels,
        upsample_ratio,
        conv_layers=4,
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
        self.conv_layers = conv_layers
        self.conv_kernel_size = conv_kernel_size
        self.convs = []

        self.upsample = TFConvTranspose1d(
            in_channels,
            kernel_size=upsample_ratio * 2,
            strides=upsample_ratio,
            padding='same',
            is_weight_norm=False,
            initializer_seed=config.initializer_seed,
        )

        self.kernel_predictor = KernelPredictor(
            cond_channels=cond_channels,
            conv_in_channels=in_channels,
            conv_out_channels=2 * in_channels,
            conv_layers=conv_layers,
            conv_kernel_size=conv_kernel_size,
            kpnet_hidden_channels=kpnet_hidden_channels,
            kpnet_conv_size=kpnet_conv_size,
            kpnet_dropout=kpnet_dropout,
        )

        for i in range(conv_layers):
            conv = tf.keras.layers.Conv1D(
                in_channels, kernel_size=conv_kernel_size, padding='same', dilation_rate=3 ** i
            )
            self.convs.append(conv)

    def call(self, x, c, **kwargs):

        _, _, in_channels = shape_list(x)
        kernels, bias = self.kernel_predictor(c)

        x = tf.keras.layers.LeakyReLU(x, alpha=0.2)
        x = self.upsample(x)

        for i in range(self.conv_layers):
            y = tf.keras.layers.LeakyReLU(x, alpha=0.2)
            y = self.convs[i](y)
            y = tf.keras.layers.LeakyReLU(y, alpha=0.2)

            k = kernels[:, i, :, :, :, :]
            b = bias[:, i, :, :]

            y = self.location_variable_convolution(y, k, b, 1, self.cond_hop_length)
            x = x + tf.nn.sigmoid(y[:, :, :in_channels]) * tf.nn.tanh(y[:, :, in_channels:])

    def location_variable_convolution(x, kernel, bias, dilation, hop_size):

        x = tf.transpose(x, [0, 2, 1])
        batch, channel, in_length = shape_list(x)
        batch, _, out_channels, kernel_size, kernel_length = shape_list(kernel)

        padding = dilation * int((kernel_size - 1) / 2)
        x = tf.pad(x, [[0, 0], [0, 0], [padding, padding]])

        step = hop_size
        size = hop_size + 2 * padding
        sizedim = in_length
        # x = x.unfold(2, hop_size + 2 * padding, hop_size)
        x = tf.reshape(x, [batch, channel, tf.cast((sizedim - size) / step + 1, tf.int32), size])
