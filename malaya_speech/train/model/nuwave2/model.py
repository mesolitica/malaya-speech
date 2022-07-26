import tensorflow.compat.v1 as tf
from math import sqrt, log, atan, exp
from ..utils import shape_list
from ..initializer import HeNormal


class DiffusionEmbedding(tf.keras.layers.Layer):
    def __init__(self, hparams, **kwargs):
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
        self.mlp_shared = tf.keras.layers.Conv1D(nhidden, kernel_size=3, padding='SAME')
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
    def __init__(self, in_channels, out_channels, bsft_channels, filter_length=1024,
                 hop_length=256,
                 win_length=1024,
                 sampling_rate=48000, **audio_kwargs):
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


class FFC(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, bsft_channels, kernel_size=3,
                 ratio_gin=0.5, ratio_gout=0.5, padding=1,
                 filter_length=1024,
                 hop_length=256,
                 win_length=1024,
                 sampling_rate=48000,
                 **audio_kwargs):
        super(FFC, self).__init__(**audio_kwargs)
        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        initializer = HeNormal()

        self.convl2l = tf.keras.layers.Conv1D(out_cl, kernel_size=kernel_size, padding='SAME',
                                              kernel_initializer=initializer,
                                              use_bias=False)
        self.convl2g = tf.keras.layers.Conv1D(out_cg, kernel_size=kernel_size, padding='SAME',
                                              kernel_initializer=initializer,
                                              use_bias=False)
        self.convg2l = tf.keras.layers.Conv1D(out_cl, kernel_size=kernel_size, padding='SAME',
                                              kernel_initializer=initializer,
                                              use_bias=False)
        self.convg2g = SpectralTransform(in_cg, out_cg, bsft_channels, **audio_kwargs)

    def call(self, x_l, x_g, band):
        out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        out_xg = self.convl2g(x_l) + self.convg2g(x_g, band)

        return out_xl, out_xg


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, residual_channels, pos_emb_dim, bsft_channels,
                 filter_length=1024,
                 hop_length=256,
                 win_length=1024,
                 sampling_rate=48000, **audio_kwargs):
        super(ResidualBlock, self).__init__(**audio_kwargs)
        initializer = HeNormal()
        self.ffc1 = FFC(residual_channels, 2*residual_channels, bsft_channels,
                        kernel_size=3, ratio_gin=0.5, ratio_gout=0.5, padding=1, **audio_kwargs)  # STFC

        self.diffusion_projection = tf.keras.layers.Dense(residual_channels)
        self.output_projection = tf.keras.layers.Conv1D(2 * residual_channels, kernel_size=1, padding='SAME',
                                                        kernel_initializer=initializer)

    def call(self, x, band, noise_level):
        noise_level = tf.expand_dims(self.diffusion_projection(noise_level), -1)
        y = x + noise_level
        y_l, y_g = y[:, :, :-self.ffc1.global_in_num], y[:, :, -self.ffc1.global_in_num:]
        y_l, y_g = self.ffc1(y_l, y_g, band)
        gate_l, filter_l = tf.split(y_l, 2, axis=-1)
        gate_g, filter_g = tf.split(y_g, 2, axis=-1)

        gate = tf.concat([gate_l, gate_g], axis=-1)
        filter = tf.concat([filter_l, filter_g], axis=-1)
        y = tf.nn.sigmoid(gate) * tf.nn.tanh(filter)
        y = self.output_projection(y)
        residual, skip = tf.split(y, 2, axis=-1)
        return (x + residual) / sqrt(2.0), skip


class NuWave2(tf.keras.layers.Layer):
    def __init__(self, hparams, **kwargs):
        super(NuWave2, self).__init__(**kwargs)
        self.hparams = hparams

        initializer = HeNormal()
        self.input_projection = tf.keras.layers.Conv1D(hparams.arch.residual_channels,
                                                       kernel_size=1, padding='SAME',
                                                       kernel_initializer=initializer)
        self.diffusion_embedding = DiffusionEmbedding(
            hparams)
        audio_kwargs = dict(filter_length=hparams.audio.filter_length, hop_length=hparams.audio.hop_length,
                            win_length=hparams.audio.win_length, sampling_rate=hparams.audio.sampling_rate)
        self.residual_layers = [
            ResidualBlock(hparams.arch.residual_channels,
                          hparams.arch.pos_emb_dim,
                          hparams.arch.bsft_channels,
                          **audio_kwargs)
            for i in range(hparams.arch.residual_layers)
        ]
        self.len_res = len(self.residual_layers)

        self.skip_projection = tf.keras.layers.Conv1D(hparams.arch.residual_channels,
                                                      kernel_size=1, padding='SAME',
                                                      kernel_initializer=initializer)

        self.output_projection = tf.keras.layers.Conv1D(1,
                                                        kernel_size=1, padding='SAME',
                                                        kernel_initializer=initializer)

    def call(self, audio, audio_low, band, noise_level):
        x = tf.stack([audio, audio_low], axis=2)
        x = self.input_projection(x)
        x = tf.nn.swish(x)
        noise_level = self.diffusion_embedding(noise_level)

        band = tf.cast(tf.one_hot(band, depth=2), tf.int32)
        skip = 0.
        for layer in self.residual_layers:
            x, skip_connection = layer(x, band, noise_level)
            # skip.append(skip_connection)
            skip += skip_connection
        x = skip / sqrt(self.len_res)
        x = self.skip_projection(x)
        x = tf.nn.swish(x)
        x = tf.squeeze(self.output_projection(x), 1)
        return x


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
