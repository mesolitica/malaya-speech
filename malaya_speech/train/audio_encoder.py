from scipy.io.wavfile import read
from scipy.signal import resample
from scipy import interpolate
import librosa
import numpy as np


def wav_to_array(file, sample_rate):
    old_samplerate, data = read(file)
    if len(data.shape) == 2:
        data = data[:, 0]

    if data.dtype not in [np.float32, np.float64]:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max

    if old_samplerate != self._sample_rate:
        old_audio = data
        duration = data.shape[0] / old_samplerate

        time_old = np.linspace(0, duration, old_audio.shape[0])
        time_new = np.linspace(
            0,
            duration,
            int(old_audio.shape[0] * self._sample_rate / old_samplerate),
        )

        interpolator = interpolate.interp1d(time_old, old_audio.T)
        data = interpolator(time_new).T

    return data.tolist(), data.shape[0] / self._sample_rate


def normalize(values):
    """
    Normalize values to mean 0 and std 1
    """
    return (values - np.mean(values)) / np.std(values)


def power_spectrogram(
    audio_data, samplerate = 22050, n_mels = 128, n_fft = 512, hop_length = 180
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
