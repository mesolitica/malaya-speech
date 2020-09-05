from scipy.io.wavfile import read
from scipy import interpolate
import librosa
import numpy as np


def normalize(values):
    return (values - np.mean(values)) / np.std(values)


def power_spectrogram(
    audio_data,
    sample_rate = 16000,
    n_mels = 128,
    n_fft = 512,
    hop_length = 180,
    normalized = True,
):
    spectrogram = librosa.feature.melspectrogram(
        audio_data,
        sr = sample_rate,
        n_mels = n_mels,
        n_fft = n_fft,
        hop_length = hop_length,
    )

    log_spectrogram = librosa.power_to_db(spectrogram, ref = np.max)
    if normalized:
        log_spectrogram = normalize(log_spectrogram)

    v = log_spectrogram.T
    return v


# https://github.com/tensorflow/models/blob/master/research/deep_speech/data/featurizer.py#L24
def spectrogram(
    samples,
    sample_rate,
    stride_ms = 10.0,
    window_ms = 20.0,
    max_freq = None,
    eps = 1e-14,
    normalized = False,
):

    if max_freq is None:
        max_freq = sample_rate / 2
    if max_freq > sample_rate / 2:
        raise ValueError(
            'max_freq must not be greater than half of sample rate.'
        )

    if stride_ms > window_ms:
        raise ValueError('Stride size must not be greater than window size.')

    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)

    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[: len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(
        samples, shape = nshape, strides = nstrides
    )
    assert np.all(
        windows[:, 1] == samples[stride_size : (stride_size + window_size)]
    )

    weighting = np.hanning(window_size)[:, None]
    fft = np.fft.rfft(windows * weighting, axis = 0)
    fft = np.absolute(fft)
    fft = fft ** 2
    scale = np.sum(weighting ** 2) * sample_rate
    fft[1:-1, :] *= 2.0 / scale
    fft[(0, -1), :] /= scale
    # Prepare fft frequency list
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])

    # Compute spectrogram feature
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    specgram = np.log(fft[:ind, :] + eps)

    specgram = np.transpose(specgram, (1, 0)).astype(np.float32)

    if normalized:
        specgram = normalize(specgram)

    return specgram


def mfcc_delta(signal, freq = 16000, n_mfcc = 5, size = 512, step = 16):
    # Mel Frequency Cepstral Coefficents
    mfcc = librosa.feature.mfcc(
        y = signal, sr = freq, n_mfcc = n_mfcc, n_fft = size, hop_length = step
    )
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order = 2)

    # Root Mean Square Energy
    mel_spectogram = librosa.feature.melspectrogram(
        y = signal, sr = freq, n_fft = size, hop_length = step
    )
    rmse = librosa.feature.rms(
        S = mel_spectogram, frame_length = size, hop_length = step
    )

    mfcc = np.asarray(mfcc)
    mfcc_delta = np.asarray(mfcc_delta)
    mfcc_delta2 = np.asarray(mfcc_delta2)
    rmse = np.asarray(rmse)

    features = np.concatenate((mfcc, mfcc_delta, mfcc_delta2, rmse), axis = 0)
    return features
