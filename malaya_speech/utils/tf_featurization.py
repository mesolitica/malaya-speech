import tensorflow as tf
import numpy as np
from tensorflow.signal import stft, inverse_stft, hann_window

separation_exponent = 2
EPSILON = 1e-10

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
