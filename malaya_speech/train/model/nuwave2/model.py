import tensorflow as tf
from math import sqrt, log, atan, exp
from ..utils import shape_list


class DiffusionEmbedding(tf.keras.layers.Layer):
    def __init__(self, hparam, **kwargs):
        super(DiffusionEmbedding, self).__init__(**kwargs)

        self.n_channels = hparams.dpm.pos_emb_channels
        self.linear_scale = hparams.dpm.pos_emb_scale
        self.out_channels = hparams.arch.pos_emb_dim

        self.projection1 = tf.keras.layers.Dense(self.out_channels)
        self.projection2 = tf.keras.layers.Dense(self.out_channels)

    def call(self, noise_level):
        if len(shape_list(noise_level)) > 1:
            noise_level = tf.squeeze(noise_level, 1)

        half_dim = self.n_channels // 2
        emb = log(10000) / (half_dim - 1)
        emb = tf.range(half_dim, dtype=tf.float32) * -emb
        emb = self.linear_scale * tf.expand_dims(noise_level, 1) * tf.expand_dims(emb, 0)
        emb = tf.concat([tf.math.sin(x), tf.math.cos(x)], axis=-1)
        x = self.projection1(x)
        x = tf.nn.swish(x)
        x = self.projection2(x)
        x = tf.nn.swish(x)
        return x


class BSFT(tf.keras.layers.Layer):
    def __init__(self, nhidden, out_channels, **kwargs):
        super(BSFT, self).__init__(**kwargs)
        initializer = HeNormal()
        self.mlp_shared = tf.keras.layers.Conv1D(hidden, kernel_size=3, padding='SAME')
        self.mlp_gamma = tf.keras.layers.Conv1D(out_channels, kernel_size=3, padding='SAME',
                                                kernel_initializer=initializer)
        self.mlp_beta = tf.keras.layers.Conv1D(out_channels, kernel_size=3, padding='SAME',
                                               kernel_initializer=initializer)

    def call(self, x, band):
        # x: (B, 2C, n_fft/2+1, T)
        # band: (B, 2, n_fft // 2 + 1) -> (B, N, n_fft // 2 + 1)

        # x tf: (B, 2C, T, n_fft/2+1)
        # band tf: (B, n_fft // 2 + 1, 2) -> (B, n_fft // 2 + 1, N)

        actv = tf.nn.swish(self.mlp_shared(band))

        # torch, (B, N, n_fft // 2 + 1) - > (B, M, n_fft // 2 + 1, 1)
        # tf, (B, n_fft // 2 + 1, N) - > (B, 1, M, n_fft // 2 + 1)
        gamma = tf.expand_dims(tf.transpose(self.mlp_gamma(actv), (0, 2, 1)), 1)
        beta = tf.expand_dims(tf.transpose(self.mlp_beta(actv), (0, 2, 1)), 1)

        # apply scale and bias
        # torch, (B, 2C, n_fft/2+1, T) * (B, M, n_fft // 2 + 1, 1)
        # tf, (B, 2C, T, n_fft/2+1) * (B, 1, M, n_fft // 2 + 1)
        out = x * (1 + gamma) + beta

        return out


class FourierUnit(tf.keras.layers.Layer):
    def __init__(self, n_channels, out_channels, bsft_channels,
                 filter_length=1024,
                 hop_length=256,
                 win_length=1024,
                 sampling_rate=48000, **kwargs):
        super(FourierUnit, self).__init__(**kwargs)

        initializer = HeNormal()
        self.sampling_rate = sampling_rate
        self.n_fft = filter_length
        self.hop_size = hop_length
        self.win_size = win_length

        self.conv_layer = tf.keras.layers.Conv2D(out_channels * 2, kernel_size=1, use_bias=False,
                                                 padding='SAME',
                                                 kernel_initializer=initializer)
        self.bsft = BSFT(bsft_channels, out_channels * 2)

    def call(self, x, band):
        batch = shape_list(x)[0]
        x = tf.reshape(x, (-1, x.shape[-1]))
        p = int((self.n_fft-self.hop_size)/2)
        padded = tf.pad(x, [[0, 0], [p, p]], mode='reflect')
        ffted_tf = tf.signal.stft(
            padded,
            self.n_fft,
            self.hop_size,
            fft_length=None,
            window_fn=tf.signal.hann_window,
            pad_end=False,
        )
        r = tf.expand_dims(tf.math.real(ffted_tf), -1)
        i = tf.expand_dims(tf.math.imag(ffted_tf), -1)
        ffted_tf = tf.concat([r, i], axis=-1)
        ffted_tf = tf.transpose(ffted_tf, (0, 3, 1, 2))
        ffted_tf_shape = shape_list(ffted_tf)
        ffted_tf = tf.reshape(ffted_tf, (batch, -1, ffted_tf_shape[2], ffted_tf_shape[3]))

        ffted_tf = tf.nn.relu(self.bsft(ffted_tf, band))
        ffted_tf = self.conv_layer(ffted_tf)

        ffted_tf = tf.reshape(ffted_tf, (-1, 2, ffted_tf_shape[2], ffted_tf_shape[3]))
        ffted_tf = tf.transpose(ffted_tf, (0, 2, 3, 1))

        ffted_tf = tf.complex(ffted_tf[:, :, :, 0], ffted_tf[:, :, :, 1])

        output = tf.signal.inverse_stft(
            ffted_tf,
            self.n_fft,
            self.hop_size,
            fft_length=None,
            window_fn=tf.signal.hann_window,
        )[:, p:-p]
        output = tf.reshape(output, (batch, -1, x.shape[-1]))
        return output


class SpectralTransform(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, bsft_channels, **audio_kwargs):
        super(SpectralTransform, self).__init__(**audio_kwargs)
        initializer = HeNormal()
        self.conv1 = tf.keras.layers.Conv1D(out_channels // 2, kernel_size=1, padding='SAME',
                                            kernel_initializer=initializer, use_bias=False)
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, bsft_channels, **audio_kwargs)
        self.conv2 = tf.keras.layers.Conv1D(out_channels, kernel_size=1, padding='SAME',
                                            kernel_initializer=initializer, use_bias=False)

    def call(self, x, band):
        x = tf.nn.swish(self.conv1(x))
        output = self.fu(x, band)
        output = self.conv2(x + output)

        return output


class Diffusion(tf.keras.layers.Layer):
    def __init__(self, hparams, **kwargs):
        super(Diffusion, self).__init__(**kwargs)

        self.hparams = hparams
        # self.model = model(hparams)

        self.logsnr_min = hparams.logsnr.logsnr_min
        self.logsnr_max = hparams.logsnr.logsnr_max

        self.logsnr_b = atan(exp(-self.logsnr_max / 2))
        self.logsnr_a = atan(exp(-self.logsnr_min / 2)) - self.logsnr_b

    def snr(self, time):
        logsnr = - 2 * tf.log(tf.tan(self.logsnr_a * time + self.logsnr_b))
        norm_nlogsnr = (self.logsnr_max - logsnr) / (self.logsnr_max - self.logsnr_min)

        alpha_sq, sigma_sq = tf.sigmoid(logsnr), tf.sigmoid(-logsnr)
        return logsnr, norm_nlogsnr, alpha_sq, sigma_sq
