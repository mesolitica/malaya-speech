import numpy as np

# https://github.com/NVIDIA/OpenSeq2Seq/blob/master/open_seq2seq/data/speech2text/speech_utils.py#L420


def mask_frequency(features, n_freq_mask: int = 2, width_freq_mask: int = 8):
    """
    Mask frequency.

    Parameters
    ----------
    features : np.array
    n_freq_mask: int, optional (default=2)
        loop size for masking.
    width_freq_mask: int, optional (default=8)
        masking size.

    Returns
    -------
    result : np.array
    """
    features = features.copy()
    for idx in range(n_freq_mask):
        freq_band = np.random.randint(width_freq_mask + 1)
        freq_base = np.random.randint(0, features.shape[1] - freq_band)
        features[:, freq_base : freq_base + freq_band] = 0
    return features


def mask_time(features, n_time_mask = 2, width_time_mask = 8):
    """
    Time frequency.

    Parameters
    ----------
    features : np.array
    n_time_mask: int, optional (default=2)
        loop size for masking.
    width_time_mask: int, optional (default=8)
        masking size.

    Returns
    -------
    result : np.array
    """
    features = features.copy()
    for idx in range(n_time_mask):
        time_band = np.random.randint(width_time_mask + 1)
        if features.shape[0] - time_band > 0:
            time_base = np.random.randint(features.shape[0] - time_band)
            features[time_base : time_base + time_band, :] = 0
    return features
