import tensorflow.compat.v1 as tf
from math import sqrt, log, atan, exp
from ..utils import shape_list
from ..initializer import HeNormal
import logging

logger = logging.getLogger(__name__)


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
        emb = tf.concat([tf.math.sin(emb), tf.math.cos(emb)], axis=-1)
        x = self.projection1(emb)
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
        # x: (B, 2C, n_fft/2+1, T), (N, C, H, W)
        # band: (B, 2, n_fft // 2 + 1) -> (B, N, n_fft // 2 + 1)

        # x tf: (B, T, n_fft/2+1, 2C), (N, H, W, C)
        # band tf: (B, n_fft // 2 + 1, N)

        actv = tf.nn.swish(self.mlp_shared(band))
        gamma = tf.expand_dims(self.mlp_gamma(actv), 1)
        beta = tf.expand_dims(self.mlp_beta(actv), 1)

        # apply scale and bias
        # torch, (B, 2C, n_fft/2+1, T) * (B, M, n_fft // 2 + 1, 1)
        # tf, (B, T, n_fft/2+1, 2C) * (B, 1, M, n_fft // 2 + 1)
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
        x = tf.transpose(x, [0, 2, 1])
        x = tf.reshape(x, (-1, tf.shape(x)[-1]))
        p = int((self.n_fft-self.hop_size)/2)
        padded = tf.pad(x, [[0, 0], [p, p]], mode='reflect')
        ffted_tf = tf.signal.stft(
            padded,
            self.n_fft,
            self.hop_size,
            fft_length=None,
            window_fn=tf.signal.hann_window,
            pad_end=False,
        )  # [BC, T, n_fft/2+1]
        ffted_tf = tf.stack([tf.math.real(ffted_tf), tf.math.imag(ffted_tf)], axis=1)
        # (B, n_fft/2+1, T, 2C)
        # (BC, 2, n_fft/2+1, T)
        ffted_tf_shape = shape_list(ffted_tf)
        ffted_tf = tf.reshape(ffted_tf, (batch, -1, ffted_tf_shape[2], ffted_tf_shape[3]))
        ffted_tf = tf.transpose(ffted_tf, [0, 2, 3, 1])

        ffted_tf = tf.nn.relu(self.bsft(ffted_tf, band))
        ffted_tf = self.conv_layer(ffted_tf)  # (B, T, n_fft/2+1, N)
        ffted_tf = tf.transpose(ffted_tf, [0, 3, 1, 2])  # (B, N, T, n_fft/2+1)

        # [BC, 2, T, n_fft/2+1]
        ffted_tf = tf.reshape(ffted_tf, (-1, 2, ffted_tf_shape[2], ffted_tf_shape[3]))
        ffted_tf = tf.complex(ffted_tf[:, 0, :, :], ffted_tf[:, 1, :, :])

        output = tf.signal.inverse_stft(
            ffted_tf,
            self.n_fft,
            self.hop_size,
            fft_length=None,
            window_fn=tf.signal.hann_window,
        )[:, p:-p]
        output = tf.reshape(output, (batch, -1, x.shape[-1]))
        output = tf.transpose(output, [0, 2, 1])
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
        noise_level = tf.expand_dims(self.diffusion_projection(noise_level), 1)
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
        self.diffusion_embedding = DiffusionEmbedding(hparams)
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

        band = tf.one_hot(band, depth=2)
        skip = 0.
        for layer in self.residual_layers:
            logger.info(f'{x}, {band}, {noise_level}')
            x, skip_connection = layer(x, band, noise_level)
            # skip.append(skip_connection)
            skip += skip_connection
        x = skip / sqrt(self.len_res)
        x = self.skip_projection(x)
        x = tf.nn.swish(x)
        x = self.output_projection(x)[:, :, 0]
        return x


class Diffusion(tf.keras.layers.Layer):
    def __init__(self, hparams, **kwargs):
        super(Diffusion, self).__init__(**kwargs)

        self.hparams = hparams
        self.model = NuWave2(hparams)

        self.logsnr_min = hparams.logsnr.logsnr_min
        self.logsnr_max = hparams.logsnr.logsnr_max

        self.logsnr_b = atan(exp(-self.logsnr_max / 2))
        self.logsnr_a = atan(exp(-self.logsnr_min / 2)) - self.logsnr_b

    def snr(self, time):
        logsnr = - 2 * tf.log(tf.tan(self.logsnr_a * time + self.logsnr_b))
        norm_nlogsnr = (self.logsnr_max - logsnr) / (self.logsnr_max - self.logsnr_min)

        alpha_sq, sigma_sq = tf.sigmoid(logsnr), tf.sigmoid(-logsnr)
        return logsnr, norm_nlogsnr, alpha_sq, sigma_sq

    def call(self, y, y_l, band, t, z=None):
        logsnr, norm_nlogsnr, alpha_sq, sigma_sq = self.snr(t)
        if z == None:
            noise = self.model(y, y_l, band, norm_nlogsnr)
        else:
            noise = z
        return noise, logsnr, (alpha_sq, sigma_sq)

    def denoise(self, y, y_l, band, t, h):
        noise, logsnr_t, (alpha_sq_t, sigma_sq_t) = self.call(y, y_l, band, t)

        f_t = - self.logsnr_a * tf.tan(self.logsnr_a * t + self.logsnr_b)
        g_t_sq = 2 * self.logsnr_a * tf.tan(self.logsnr_a * t + self.logsnr_b)

        dzt_det = (f_t * y - 0.5 * g_t_sq * (-noise / tf.sqrt(sigma_sq_t))) * h

        denoised = y - dzt_det
        return denoised

    def denoise_ddim(self, y, y_l, band, logsnr_t, logsnr_s, z=None):
        norm_nlogsnr = (self.logsnr_max - logsnr_t) / (self.logsnr_max - self.logsnr_min)

        alpha_sq_t, sigma_sq_t = tf.sigmoid(logsnr_t), tf.sigmoid(-logsnr_t)

        if z == None:
            noise = self.model(y, y_l, band, norm_nlogsnr)
        else:
            noise = z

        alpha_sq_s, sigma_sq_s = tf.sigmoid(logsnr_s), tf.sigmoid(-logsnr_s)

        pred = (y - tf.sqrt(sigma_sq_t) * noise) / tf.sqrt(alpha_sq_t)

        denoised = tf.sqrt(alpha_sq_s) * pred + tf.sqrt(sigma_sq_s) * noise
        return denoised, pred

    def diffusion(self, signal, noise, s, t=None):
        bsize = tf.shape(s)[0]

        time = s if t is None else tf.concat([s, t], axis=0)

        _, _, alpha_sq, sigma_sq = self.snr(time)
        if t is not None:
            alpha_sq_s, alpha_sq_t = alpha_sq[:bsize], alpha_sq[bsize:]
            sigma_sq_s, sigma_sq_t = sigma_sq[:bsize], sigma_sq[bsize:]

            alpha_sq_tbars = alpha_sq_t / alpha_sq_s
            sigma_sq_tbars = sigma_sq_t - alpha_sq_tbars * sigma_sq_s

            alpha_sq, sigma_sq = alpha_sq_tbars, sigma_sq_tbars

        alpha = tf.sqrt(alpha_sq)
        sigma = tf.sqrt(sigma_sq)

        noised = tf.expand_dims(alpha, -1) * signal + tf.expand_dims(sigma, -1) * noise
        return alpha, sigma, noised


class Model(tf.keras.Model):
    def __init__(self, hparams, **kwargs):
        super().__init__(**kwargs)

        self.model = Diffusion(hparams)

    def call(self, wav, wav_l, band, t):
        z = tf.random.normal(tf.shape(wav))
        _, _, diffusion = self.model.diffusion(wav, z, t)

        estim, logsnr, _ = self.model(diffusion, wav_l, band, t)
        return estim, z, logsnr, wav, diffusion, logsnr

    def common_step(self, wav, wav_l, band, t):
        noise_estimation, z, logsnr, wav, wav_noisy, logsnr = self(wav, wav_l, band, t)

        loss = self.loss(noise_estimation, z)
        return loss, wav, wav_noisy, z, noise_estimation, logsnr

    def inference(self, wav_l, band, step, noise_schedule=None):
        signal = tf.random.normal(tf.shape(wav_l))
        signal_list = []
        for i in range(step):
            if noise_schedule == None:
                t = (1.0 - (i+0.5) * 1.0/step) * tf.ones(shape=(tf.shape(signal)[0],))
                signal = self.model.denoise(signal, wav_l, band, t, 1.0/step)
            else:
                logsnr_t = noise_schedule[i] * tf.ones(shape=(tf.shape(signal)[0],))
                if i == step-1:
                    logsnr_s = self.hparams.logsnr.logsnr_max * tf.ones(shape=(tf.shape(signal)[0],))
                else:
                    logsnr_s = noise_schedule[i+1] * tf.ones(shape=(tf.shape(signal)[0],))
                signal, recon = self.model.denoise_ddim(signal, wav_l, band, logsnr_t, logsnr_s)
            signal_list.append(signal)

        wav_recon = tf.clip_by_value(signal, -1, 1)
        return wav_recon, signal_list
