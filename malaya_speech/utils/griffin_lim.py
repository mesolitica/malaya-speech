import librosa
import numpy as np


def from_mel(
    mel_,
    sr = 16000,
    n_fft = 2048,
    n_iter = 32,
    win_length = 1000,
    hop_length = 100,
):
    """
    Change melspectrogram into waveform using Librosa.

    Parameters
    ----------
    spectrogram: np.array

    Returns
    --------
    result: np.array
    """
    return librosa.feature.inverse.mel_to_audio(
        mel_,
        sr = sr,
        n_fft = n_fft,
        hop_length = hop_length,
        win_length = win_length,
        window = 'hann',
        center = True,
        pad_mode = 'reflect',
        power = 1.0,
        n_iter = n_iter,
    )


def from_mel_vocoder(
    mel,
    sr = 22050,
    n_fft = 1024,
    n_mels = 256,
    fmin = 80,
    fmax = 7600,
    n_iter = 32,
    win_length = None,
    hop_length = 256,
):
    """
    Change melspectrogram into waveform using Librosa.

    Parameters
    ----------
    spectrogram: np.array

    Returns
    --------
    result: np.array
    """
    mel_spec = np.power(10.0, mel)
    mel_basis = librosa.filters.mel(
        sr, n_fft = n_fft, n_mels = n_mels, fmin = fmin, fmax = fmax
    )
    mel_to_linear = np.maximum(
        1e-10, np.dot(np.linalg.pinv(mel_basis), mel_spec)
    )
    gl_lb = librosa.griffinlim(
        mel_to_linear,
        n_iter = n_iter,
        hop_length = hop_length,
        win_length = win_length,
    )
    return gl_lb
