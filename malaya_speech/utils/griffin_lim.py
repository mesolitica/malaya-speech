import librosa
import numpy as np


def from_spectrogram(spect, n_iter = 32, win_length = 1000, hop_length = 100):
    """
    Change spectrogram into waveform using Librosa.

    Parameters
    ----------
    spectrogram: np.array

    Returns
    --------
    result: np.array
    """
    return librosa.griffinlim(
        spectrogram,
        n_iter = n_iter,
        win_length = win_length,
        hop_length = hop_length,
    )


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
        n_iter = 32,
    )
