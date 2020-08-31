from scipy.io.wavfile import read
from scipy.signal import resample
from scipy import interpolate
import librosa
import numpy as np
import soundfile as sf


def change_samplerate(data, old_samplerate, new_samplerate):
    old_audio = data
    duration = data.shape[0] / old_samplerate
    time_old = np.linspace(0, duration, old_audio.shape[0])
    time_new = np.linspace(
        0, duration, int(old_audio.shape[0] * new_samplerate / old_samplerate)
    )

    interpolator = interpolate.interp1d(time_old, old_audio.T)
    data = interpolator(time_new).T
    return data


def read_audio(data, old_samplerate, sample_rate = 16000):
    if len(data.shape) == 2:
        data = data[:, 0]
    if data.dtype not in [np.float32, np.float64]:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max

    if old_samplerate != sample_rate:
        data = change_samplerate(data, old_samplerate, sample_rate)

    return data.tolist(), data.shape[0] / sample_rate


def wav_to_array(file, sample_rate = 16000):
    old_samplerate, data = read(file)
    return read_audio(data, old_samplerate, sample_rate)


def flac_to_array(file, sample_rate = 16000):
    data, old_samplerate = sf.read(file)
    return read_audio(data, old_samplerate, sample_rate)


def normalize(values):
    return (values - np.mean(values)) / np.std(values)


def power_spectrogram(
    audio_data, samplerate = 16000, n_mels = 128, n_fft = 512, hop_length = 180
):
    spectrogram = librosa.feature.melspectrogram(
        audio_data,
        sr = samplerate,
        n_mels = n_mels,
        n_fft = n_fft,
        hop_length = hop_length,
    )

    log_spectrogram = librosa.power_to_db(spectrogram, ref = np.max)
    normalized_spectrogram = normalize(log_spectrogram)

    v = normalized_spectrogram.T
    return v


# https://github.com/tensorflow/models/blob/master/research/deep_speech/data/featurizer.py#L24
def spectrogram(
    samples,
    sample_rate,
    stride_ms = 10.0,
    window_ms = 20.0,
    max_freq = None,
    eps = 1e-14,
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
    return np.transpose(specgram, (1, 0))
