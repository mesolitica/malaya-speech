import tensorflow as tf
import numpy as np
import math
import librosa
from tensorflow.signal import stft, inverse_stft, hann_window
from malaya_speech.utils.constant import ECAPA_TDNN_WINDOWS

separation_exponent = 2
EPSILON = 1e-10


class ECAPA_TCNNFeaturizer:
    def __init__(
        self,
        sample_rate = 16000,
        win_length = 25,
        hop_length = 10,
        n_fft = 400,
        n_mels = 80,
        log_mel = True,
        f_min = 0,
        f_max = 8000,
        power_spectrogram = 2,
        amin = 1e-10,
        ref_value = 1.0,
        top_db = 80.0,
        param_change_factor = 1.0,
        param_rand_factor = 0.0,
        window_length = 5,
        **kwargs,
    ):
        self.sample_rate = sample_rate
        self.win_length = int(round((sample_rate / 1000.0) * win_length))
        self.hop_length = int(round((sample_rate / 1000.0) * hop_length))
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.log_mel = log_mel
        self.f_min = f_min
        self.f_max = f_max
        self.n_stft = self.n_fft // 2 + 1
        self.amin = amin
        self.ref_value = ref_value
        self.db_multiplier = math.log10(max(self.amin, self.ref_value))
        self.power_spectrogram = power_spectrogram
        self.top_db = top_db

        if self.power_spectrogram == 2:
            self.multiplier = 10
        else:
            self.multiplier = 20

        mel = np.linspace(
            self._to_mel(self.f_min), self._to_mel(self.f_max), self.n_mels + 2
        )
        hz = self._to_hz(mel)

        band = hz[1:] - hz[:-1]
        self.band = band[:-1]
        self.f_central = hz[1:-1]

        all_freqs = np.linspace(0, self.sample_rate // 2, self.n_stft)
        all_freqs = np.expand_dims(all_freqs, 0)
        self.all_freqs_mat = np.tile(all_freqs, (self.f_central.shape[0], 1))

        self.n = (window_length - 1) // 2
        self.denom = self.n * (self.n + 1) * (2 * self.n + 1) / 3
        a = np.arange(-self.n, self.n + 1, dtype = np.float32)
        a = np.expand_dims(np.expand_dims(a, 0), 0)
        self.kernel = np.tile(a, (self.n_mels, 1, 1))

        if not tf.executing_eagerly():
            with tf.device('/cpu:0'):
                self._X = tf.compat.v1.placeholder(tf.float32, (None, None, 1))
                self._K = tf.compat.v1.placeholder(tf.float32, (None, 1, 1))
                self._conv = tf.nn.conv1d(
                    self._X, self._K, 1, padding = 'VALID'
                )
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self._sess = tf.compat.v1.Session(config = config)

    def _to_hz(self, mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def _to_mel(self, hz):
        return 2595 * math.log10(1 + hz / 700)

    def _triangular_filters(self, all_freqs, f_central, band):
        slope = (all_freqs - f_central) / band
        left_side = slope + 1.0
        right_side = -slope + 1.0
        zero = np.zeros(1)
        fbank_matrix = np.maximum(zero, np.minimum(left_side, right_side)).T

        return fbank_matrix

    def _amplitude_to_DB(self, x):
        x_db = self.multiplier * np.log10(
            np.clip(x, a_min = self.amin, a_max = None)
        )
        x_db -= self.multiplier * self.db_multiplier
        new_x_db_max = x_db.max() - self.top_db
        x_db = np.maximum(x_db, new_x_db_max)
        return x_db

    def _group_conv(self, x, kernel):
        x = x.astype(np.float32)
        kernel = kernel.copy().astype(np.float32)
        p = []
        for i in range(self.n_mels):
            if tf.executing_eagerly():
                c = tf.nn.conv1d(
                    x[:, :, i : i + 1],
                    kernel[:, :, i : i + 1],
                    1,
                    padding = 'VALID',
                )
            else:
                c = self._sess.run(
                    self._conv,
                    feed_dict = {
                        self._X: x[:, :, i : i + 1],
                        self._K: kernel[:, :, i : i + 1],
                    },
                )
            p.append(c)

        return np.concatenate(p, axis = 2)

    def vectorize(self, signal):
        s = librosa.stft(
            y,
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            win_length = self.win_length,
            window = ECAPA_TDNN_WINDOWS,
            pad_mode = 'constant',
        )
        s = np.concatenate(
            [np.expand_dims(s.real, -1), np.expand_dims(s.imag, -1)], -1
        )
        s = np.transpose(s, (1, 0, 2))
        f_central_mat = np.tile(
            np.expand_dims(self.f_central, 0), (self.all_freqs_mat.shape[1], 1)
        ).T
        band_mat = np.tile(
            np.expand_dims(self.band, 0), (self.all_freqs_mat.shape[1], 1)
        ).T
        fbank_matrix = self._triangular_filters(
            self.all_freqs_mat, f_central_mat, band_mat
        )
        s = np.expand_dims(s, 0)
        sp_shape = s.shape
        s = s.reshape(sp_shape[0] * sp_shape[3], sp_shape[1], sp_shape[2])
        fbanks = np.einsum('ijk,kl->ijl', s, fbank_matrix)
        fbanks = self._amplitude_to_DB(fbanks)
        fb_shape = fbanks.shape
        fbanks = fbanks.reshape(
            sp_shape[0], fb_shape[1], fb_shape[2], sp_shape[3]
        )
        x = np.transpose(fbanks, (0, 2, 3, 1))
        or_shape = x.shape
        if len(or_shape) == 4:
            x = x.reshape(or_shape[0] * or_shape[2], or_shape[1], or_shape[3])
        x = np.pad(x, ((0, 0), (0, 0), (self.n, self.n)), mode = 'edge')
        x = np.transpose(x, (0, 2, 1))
        k = np.transpose(self.kernel, (2, 1, 0))
        conv = self._group_conv(x, k)
        conv = np.transpose(conv, (0, 2, 1))
        delta_coeff = conv / self.denom
        if len(or_shape) == 4:
            delta_coeff = delta_coeff.reshape(
                or_shape[0], or_shape[1], or_shape[2], or_shape[3]
            )
        delta_coeff = np.transpose(delta_coeff, (0, 3, 1, 2))
        return delta_coeff


# https://github.com/TensorSpeech/TensorFlowASR/blob/main/tensorflow_asr/featurizers/speech_featurizers.py#L370
class STTFeaturizer:
    def __init__(
        self,
        sample_rate = 16000,
        frame_ms = 25,
        stride_ms = 10,
        num_feature_bins = 80,
        feature_type = 'log_mel_spectrogram',
        preemphasis = 0.97,
        dither = 1e-5,
        normalize_signal = True,
        normalize_feature = True,
        normalize_per_feature = False,
        **kwargs,
    ):
        self.sample_rate = sample_rate
        self.frame_length = int(self.sample_rate * (frame_ms / 1000))
        self.frame_step = int(self.sample_rate * (stride_ms / 1000))
        self.num_feature_bins = num_feature_bins
        self.feature_type = feature_type
        self.preemphasis = preemphasis
        self.dither = dither
        self.normalize_signal = normalize_signal
        self.normalize_feature = normalize_feature
        self.normalize_per_feature = normalize_per_feature

    @property
    def nfft(self) -> int:
        return 2 ** (self.frame_length - 1).bit_length()

    def __call__(self, signal):
        return self.vectorize(signal)

    def vectorize(self, signal):
        if self.normalize_signal:
            signal = normalize_signal(signal)
        if self.dither > 0:
            signal += self.dither * tf.random.normal(shape = tf.shape(signal))
        signal = preemphasis(signal, self.preemphasis)
        if self.feature_type == 'mfcc':
            features = self.compute_mfcc(signal)
        elif self.feature_type == 'log_mel_spectrogram':
            features = self.compute_log_mel_spectrogram(signal)
        elif self.feature_type == 'spectrogram':
            features = self.compute_spectrogram(signal)
        else:
            raise ValueError(
                "feature_type must be either 'mfcc', "
                "'log_mel_spectrogram', or 'spectrogram'"
            )

        if self.normalize_feature:
            features = normalize_audio_features(
                features, per_feature = self.normalize_per_feature
            )
        return features

    def stft(self, signal):
        return tf.square(
            tf.abs(
                tf.signal.stft(
                    signal,
                    frame_length = self.frame_length,
                    frame_step = self.frame_step,
                    fft_length = self.nfft,
                )
            )
        )

    def power_to_db(self, S, ref = 1.0, amin = 1e-10, top_db = 80.0):
        if amin <= 0:
            raise ValueError('amin must be strictly positive')

        magnitude = S

        ref_value = np.abs(ref)
        log_spec = 10.0 * log10(tf.maximum(amin, magnitude))
        log_spec -= 10.0 * log10(tf.maximum(amin, ref_value))
        if top_db is not None:
            if top_db < 0:
                raise ValueError('top_db must be non-negative')
            log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

        return log_spec

    def compute_log_mel_spectrogram(self, signal):
        spectrogram = self.stft(signal)
        linear_to_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins = self.num_feature_bins,
            num_spectrogram_bins = spectrogram.shape[-1],
            sample_rate = self.sample_rate,
            lower_edge_hertz = 0.0,
            upper_edge_hertz = (self.sample_rate / 2),
        )
        mel = tf.tensordot(spectrogram, linear_to_weight_matrix, 1)
        return tf.math.log(mel + 1e-20)

    def compute_spectrogram(self, signal):
        S = self.stft(signal)
        spectrogram = self.power_to_db(S)
        return spectrogram[:, : self.num_feature_bins]

    def compute_mfcc(self, signal):
        log_mel_spectrogram = self.compute_log_mel_spectrogram(signal)
        return tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)


def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype = numerator.dtype))
    return numerator / denominator


def normalize_audio_features(audio_feature, per_feature = False):
    axis = 0 if per_feature else None
    mean = tf.reduce_mean(audio_feature, axis = axis)
    std_dev = tf.math.reduce_std(audio_feature, axis = axis) + 1e-9
    return (audio_feature - mean) / std_dev


def normalize_signal(signal):
    gain = 1.0 / (tf.reduce_max(tf.abs(signal), axis = -1) + 1e-9)
    return signal * gain


def preemphasis(signal, coeff = 0.97):
    if not coeff or coeff <= 0.0:
        return signal
    s0 = tf.expand_dims(signal[0], axis = -1)
    s1 = signal[1:] - coeff * signal[:-1]
    return tf.concat([s0, s1], axis = -1)


def pad_and_partition(tensor, segment_len):
    tensor_size = tf.math.floormod(tf.shape(tensor)[0], segment_len)
    pad_size = tf.math.floormod(segment_len - tensor_size, segment_len)
    padded = tf.pad(
        tensor, [[0, pad_size]] + [[0, 0]] * (len(tensor.shape) - 1)
    )
    split = (tf.shape(padded)[0] + segment_len - 1) // segment_len
    return tf.reshape(
        padded,
        tf.concat([[split, segment_len], tf.shape(padded)[1:]], axis = 0),
    )


def pad_and_reshape(
    instr_spec, frame_length, frame_step = 1024, T = 512, F = 1024
):
    spec_shape = tf.shape(instr_spec)
    extension_row = tf.zeros((spec_shape[0], spec_shape[1], 1, spec_shape[-1]))
    n_extra_row = (frame_length) // 2 + 1 - F
    extension = tf.tile(extension_row, [1, 1, n_extra_row, 1])
    extended_spec = tf.concat([instr_spec, extension], axis = 2)
    old_shape = tf.shape(extended_spec)
    new_shape = tf.concat(
        [[old_shape[0] * old_shape[1]], old_shape[2:]], axis = 0
    )
    processed_instr_spec = tf.reshape(extended_spec, new_shape)
    return processed_instr_spec


def extend_mask(
    mask,
    extension = 'zeros',
    frame_length = 4096,
    frame_step = 1024,
    T = 512,
    F = 1024,
):
    if extension == 'average':
        extension_row = tf.reduce_mean(mask, axis = 2, keepdims = True)
    elif extension == 'zeros':
        mask_shape = tf.shape(mask)
        extension_row = tf.zeros(
            (mask_shape[0], mask_shape[1], 1, mask_shape[-1])
        )
    else:
        raise ValueError(f'Invalid mask_extension parameter {extension}')
    n_extra_row = frame_length // 2 + 1 - F
    extension = tf.tile(extension_row, [1, 1, n_extra_row, 1])
    return tf.concat([mask, extension], axis = 2)


def get_stft(
    y,
    return_magnitude = True,
    frame_length = 4096,
    frame_step = 1024,
    T = 512,
    F = 1024,
):

    waveform = tf.concat(
        [tf.zeros((frame_length, 1)), tf.expand_dims(y, -1)], 0
    )
    stft_feature = tf.transpose(
        stft(
            tf.transpose(waveform),
            frame_length,
            frame_step,
            window_fn = lambda frame_length, dtype: (
                hann_window(frame_length, periodic = True, dtype = dtype)
            ),
            pad_end = True,
        ),
        perm = [1, 2, 0],
    )
    if return_magnitude:
        D = tf.abs(pad_and_partition(stft_feature, T))[:, :, :F, :]
        return stft_feature, D
    else:
        return stft_feature


def istft(
    stft_t,
    y,
    time_crop = None,
    factor = 2 / 3,
    frame_length = 4096,
    frame_step = 1024,
    T = 512,
    F = 1024,
):

    inversed = (
        inverse_stft(
            tf.transpose(stft_t, perm = [2, 0, 1]),
            frame_length,
            frame_step,
            window_fn = lambda frame_length, dtype: (
                hann_window(frame_length, periodic = True, dtype = dtype)
            ),
        )
        * factor
    )
    reshaped = tf.transpose(inversed)
    if time_crop is None:
        time_crop = tf.shape(y)[0]
    return reshaped[frame_length : frame_length + time_crop, :]
