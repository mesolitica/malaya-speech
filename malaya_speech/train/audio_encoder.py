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
