import tensorflow as tf
import numpy as np
from ..utils import shape_list
from ..melgan.layer import WeightNormalization
from ..universal_melgan.model import TFConvTranspose1d


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
            WeightNormalization(tf.keras.layers.Conv1D(
                filters=kpnet_hidden_channels,
                kernel_size=5,
                use_bias=True,
                padding='same',
            )),
            getattr(tf.keras.layers, kpnet_nonlinear_activation)(
                **kpnet_nonlinear_activation_params
            )])

        padding = (kpnet_conv_size - 1) // 2
        self.residual_convs = []
        for _ in range(3):
            s = tf.keras.models.Sequential([
                tf.keras.layers.Dropout(kpnet_dropout),
                WeightNormalization(tf.keras.layers.Conv1D(kpnet_hidden_channels,
                                    kpnet_conv_size, padding='same', use_bias=True)),
                getattr(tf.keras.layers, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
                WeightNormalization(tf.keras.layers.Conv1D(kpnet_hidden_channels,
                                    kpnet_conv_size, padding='same', use_bias=True)),
                getattr(tf.keras.layers, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
            ])
            self.residual_convs.append(s)

        self.kernel_conv = WeightNormalization(tf.keras.layers.Conv1D(
            l_w, kpnet_conv_size, padding='same', use_bias=True))
        self.bias_conv = WeightNormalization(tf.keras.layers.Conv1D(
            l_b, kpnet_conv_size, padding='same', use_bias=True))

    def call(self, c, training=True):
        batch, cond_length, _ = shape_list(c)

        c = self.input_conv(c)
        for residual_conv in self.residual_convs:
            c = c + residual_conv(c, training=training)
        k = self.kernel_conv(c)
        b = self.bias_conv(c)
        kernels = tf.reshape(k, (batch, cond_length, self.conv_layers, self.conv_in_channels,
                                 self.conv_out_channels, self.conv_kernel_size))
        bias = tf.reshape(b, (batch, cond_length, self.conv_layers, self.conv_out_channels))
        return kernels, bias


class LVCBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        in_channels,
        cond_channels,
        stride,
        dilations=[1, 3, 9, 27],
        lReLU_slope=0.2,
        conv_kernel_size=3,
        cond_hop_length=256,
        kpnet_hidden_channels=64,
        kpnet_conv_size=3,
        kpnet_dropout=0.0,
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
            kpnet_nonlinear_activation_params={'alpha': lReLU_slope}
        )

        self.convt_pre = tf.keras.models.Sequential([
            tf.keras.layers.LeakyReLU(alpha=lReLU_slope),
            TFConvTranspose1d(
                in_channels,
                kernel_size=stride * 2,
                strides=stride,
                padding='same',
                is_weight_norm=True,
                initializer_seed=42,
            )])

        self.convs = []
        for d in dilations:
            conv = tf.keras.models.Sequential([
                tf.keras.layers.LeakyReLU(alpha=lReLU_slope),
                WeightNormalization(tf.keras.layers.Conv1D(
                    in_channels, kernel_size=conv_kernel_size, padding='same', dilation_rate=d
                )),
                tf.keras.layers.LeakyReLU(alpha=lReLU_slope),
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

            k = kernels[:, :, i, :, :, :]
            b = bias[:, :, i, :]

            output = self.location_variable_convolution(output, k, b, hop_size=self.cond_hop_length)
            x = x + tf.nn.sigmoid(output[:, :, :in_channels]) * tf.nn.tanh(output[:, :, in_channels:])

        return x

    def location_variable_convolution(self, x, kernel, bias, dilation=1, hop_size=256):
        """
        x = [batch, in_length, channel]
        kernel = [batch, kernel_length, in_channels, out_channels, kernel_size]
        bias = [batch, kernel_length, out_channels]
        """
        x = tf.transpose(x, [0, 2, 1])
        kernel = tf.transpose(kernel, [0, 2, 3, 4, 1])
        bias = tf.transpose(bias, [0, 2, 1])

        batch, channel, in_length = shape_list(x)
        batch, _, out_channels, kernel_size, kernel_length = shape_list(kernel)

        padding = dilation * int((kernel_size - 1) / 2)
        x = tf.pad(x, ((0, 0), (0, 0), (padding, padding)))

        dim_size = tf.shape(x)[2]
        size = hop_size + 2 * padding
        step = hop_size
        size_output = (dim_size - size) // step + 1
        features = tf.TensorArray(dtype=tf.float32, size=size_output, dynamic_size=True, infer_shape=False)
        init_state = (0, features)

        def condition(i, features):
            return i < size_output

        def body(i, features):
            return i + 1, features.write(i, x[:, :, i: i + size])

        _, features = tf.while_loop(condition, body, init_state)
        x = features.stack()
        x.set_shape((None, None, channel, size))  # [stacked, b, channel, size]
        x = tf.transpose(x, [1, 2, 0, 3])

        dim_size = tf.shape(x)[3]
        size = dilation
        step = dilation
        size_output = (dim_size - size) // step + 1
        features = tf.TensorArray(dtype=tf.float32, size=size_output, dynamic_size=True, infer_shape=False)
        init_state = (0, features)

        def condition(i, features):
            return i < size_output

        def body(i, features):
            return i + 1, features.write(i, x[:, :, :, i: i + size])

        _, features = tf.while_loop(condition, body, init_state)
        x = features.stack()
        x.set_shape((None, None, channel, None, size))
        x = tf.transpose(x, [1, 2, 3, 0, 4])

        x = x[:, :, :, :, :hop_size]
        x = tf.transpose(x, [0, 1, 2, 4, 3])

        dim_size = tf.shape(x)[4]
        size = kernel_size
        step = dilation
        size_output = (dim_size - size) // step + 1
        features = tf.TensorArray(dtype=tf.float32, size=size_output, dynamic_size=True, infer_shape=False)
        init_state = (0, features)

        def condition(i, features):
            return i < size_output

        def body(i, features):
            return i + 1, features.write(i, x[:, :, :, :, i: i + size])

        _, features = tf.while_loop(condition, body, init_state)
        x = features.stack()
        x.set_shape((None, None, channel, None, dilation, size))
        x = tf.transpose(x, [1, 2, 3, 4, 0, 5])

        o = tf.einsum('bildsk,biokl->bolsd', x, kernel)
        o = o + tf.expand_dims(tf.expand_dims(bias, -1), -1)
        o = tf.reshape(o, [batch, out_channels, -1])
        o = tf.transpose(o, [0, 2, 1])
        return o
