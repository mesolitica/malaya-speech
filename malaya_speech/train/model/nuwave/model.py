import tensorflow as tf
from math import sqrt
from ..initializer import HeNormal


class DiffusionEmbedding(tf.keras.layers.Layer):
    def __init__(self, hparam, **kwargs):
        super(DiffusionEmbedding, self).__init__(**kwargs)
        self.n_channels = hparam.ddpm.pos_emb_channels
        self.scale = hparam.ddpm.pos_emb_scale
        self.out_channels = hparam.arch.pos_emb_dim
        half_dim = self.n_channels // 2
        exponents = tf.range(half_dim) / tf.cast(half_dim, tf.float32)
        self.exponents = 1e-4 ** exponents
        self.projection1 = tf.keras.layers.Dense(self.out_channels)
        self.projection2 = tf.keras.layers.Dense(self.out_channels)

    def call(self, noise_level):
        x = self.scale * noise_level * tf.expand_dims(self.exponents, 0)
        x = tf.concat([tf.math.sin(x), tf.math.cos(x)], axis=-1)
        x = self.projection1(x)
        x = tf.nn.swish(x)
        x = self.projection2(x)
        x = tf.nn.swish(x)
        return x


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, residual_channels, dilation, pos_emb_dim, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        initializer = HeNormal()
        self.dilated_conv = tf.keras.layers.Conv1D(2*residual_channels, 3,
                                                   padding='SAME',
                                                   dilation_rate=dilation,
                                                   kernel_initializer=initializer)
        self.diffusion_projection = tf.keras.layers.Dense(residual_channels)
        self.output_projection = tf.keras.layers.Conv1D(2 * residual_channels, 1,
                                                        kernel_initializer=initializer, padding='SAME')
        self.low_projection = tf.keras.layers.Conv1D(2*residual_channels, 3,
                                                     padding='SAME',
                                                     dilation_rate=dilation,
                                                     kernel_initializer=initializer)

    def call(self, x, x_low, noise_level):
        noise_level = tf.expand_dims(self.diffusion_projection(noise_level), -1)
        y = x + noise_level
        y = self.dilated_conv(y)
        y += self.low_projection(x_low)
        gate, filter = tf.split(y, 2, axis=-1)
        y = tf.nn.sigmoid(gate) * tf.nn.tanh(filter)
        y = self.output_projection(y)
        residual, skip = tf.split(y, 2, axis=-1)
        return (x + residual) / sqrt(2.0), skip


class Model(tf.keras.Model):
    def __init__(self, hparams, **kwargs):
        super().__init__()
        initializer = HeNormal()
        self.hparams = hparams
        self.input_projection = tf.keras.layers.Conv1D(hparams.arch.residual_channels, 1)
        self.low_projection = tf.keras.layers.Conv1D(hparams.arch.residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(hparams)
        self.residual_layers = [
            ResidualBlock(hparams.arch.residual_channels,
                          2**(i % hparams.arch.dilation_cycle_length),
                          hparams.arch.pos_emb_dim)
            for i in range(hparams.arch.residual_layers)
        ]
        self.len_res = len(self.residual_layers)
        self.skip_projection = tf.keras.layers.Conv1D(hparams.arch.residual_channels, 1)
        self.output_projection = tf.keras.layers.Conv1D(1, 1, kernel_initializer=initializer)

    def call(self, audio, audio_low, noise_level):
        x = tf.expand_dims(audio, 1)
        x = self.input_projection(x)
        x = silu(x)
        x_low = self.low_projection(tf.expand_dims(audio, 1))
        x_low = silu(x_low)
        noise_level = self.diffusion_embedding(noise_level)

        skip = 0.
        for layer in self.residual_layers:
            x, skip_connection = layer(x, x_low, noise_level)
            # skip.append(skip_connection)
            skip += skip_connection

        x = skip / sqrt(self.len_res)
        x = self.skip_projection(x)
        x = silu(x)
        x = self.output_projection(x)[:, :, 0]
        return x
