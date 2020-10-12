import librosa
import numpy as np


def from_mel(
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
