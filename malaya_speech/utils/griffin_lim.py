import librosa as librosa_
import numpy as np


def librosa(spectrogram, n_iter = 100, win_length = 1000, hop_length = 200):
    return librosa_.griffinlim(
        spectrogram,
        n_iter = n_iter,
        win_length = win_length,
        hop_length = hop_length,
    )
