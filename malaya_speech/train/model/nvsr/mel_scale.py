import tensorflow as tf
import math


def _hz_to_mel(freq: float, mel_scale: str = "htk"):
    if mel_scale not in ["slaney", "htk"]:
        raise ValueError('mel_scale should be one of "htk" or "slaney".')

    if mel_scale == "htk":
        return 2595.0 * math.log10(1.0 + (freq / 700.0))

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (freq - f_min) / f_sp

    # Fill in the log-scale part
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    if freq >= min_log_hz:
        mels = min_log_mel + math.log(freq / min_log_hz) / logstep

    return mels


def _mel_to_hz(mels, mel_scale: str = "htk"):
    if mel_scale not in ["slaney", "htk"]:
        raise ValueError('mel_scale should be one of "htk" or "slaney".')
    if mel_scale == "htk":
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    log_t = mels >= min_log_mel
    indices = tf.where(log_t)
    outputs_ = min_log_hz * tf.exp(logstep * (mels[log_t] - min_log_mel))
    return tf.tensor_scatter_nd_update(
        freqs, tf.where(log_t), outputs_, name=None
    )


def _create_triangular_filterbank(all_freqs, f_pts):
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = tf.expand_dims(f_pts, 0) - tf.expand_dims(all_freqs, 1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    zero = tf.zeros(shape=tf.shape(down_slopes))
    fb = tf.maximum(zero, tf.minimum(down_slopes, up_slopes))
    return fb


def melscale_fbanks(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    sample_rate: int,
    norm=None,
    mel_scale: str = "htk",
):
    all_freqs = tf.cast(tf.linspace(0, sample_rate // 2, n_freqs), tf.float32)
    m_min = _hz_to_mel(f_min, mel_scale=mel_scale)
    m_max = _hz_to_mel(f_max, mel_scale=mel_scale)
    m_pts = tf.cast(tf.linspace(m_min, m_max, n_mels + 2), tf.float32)
    f_pts = _mel_to_hz(m_pts, mel_scale=mel_scale)
    fb = _create_triangular_filterbank(all_freqs, f_pts)
    if norm is not None and norm == "slaney":
        enorm = 2.0 / (f_pts[2: n_mels + 2] - f_pts[:n_mels])
        fb *= tf.expand_dims(enorm, 0)

    return fb


class MelScale(tf.keras.layers.Layer):
    def __init__(
        self,
        n_mels: int = 128,
        sample_rate: int = 16000,
        f_min: float = 0.0,
        f_max=None,
        n_stft: int = 201,
        norm=None,
        mel_scale: str = "htk",
        **kwargs,
    ):
        super(MelScale, self).__init__(**kwargs)

        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min
        self.norm = norm
        self.mel_scale = mel_scale

        self.fb = melscale_fbanks(
            n_stft,
            self.f_min,
            self.f_max,
            self.n_mels,
            self.sample_rate,
            self.norm,
            self.mel_scale,
        )

    def call(self, specgram):
        return tf.matmul(specgram, fb)
