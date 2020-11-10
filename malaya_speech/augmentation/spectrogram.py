import numpy as np
import tensorflow as tf

# https://github.com/NVIDIA/OpenSeq2Seq/blob/master/open_seq2seq/data/speech2text/speech_utils.py#L420
# https://github.com/Kyubyong/specAugment/blob/master/USER_DIR/speech_recognition.py


def mask_frequency(
    features, n_freq_mask: int = 2, width_freq_mask: int = 8, random_band = True
):
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
        if random_band:
            freq_band = np.random.randint(width_freq_mask + 1)
        else:
            freq_band = width_freq_mask
        freq_base = np.random.randint(0, features.shape[1] - freq_band)
        features[:, freq_base : freq_base + freq_band] = 0
    return features


def mask_time(
    features, n_time_mask = 2, width_time_mask = 8, random_band = True
):
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
        if random_band:
            time_band = np.random.randint(width_time_mask + 1)
        else:
            time_band = width_time_mask
        if features.shape[0] - time_band > 0:
            time_base = np.random.randint(features.shape[0] - time_band)
            features[time_base : time_base + time_band, :] = 0
    return features


def tf_mask_frequency(features, F = 27):
    """
    Mask frequency using Tensorflow.

    Parameters
    ----------

    features : np.array
    F: size of mask for frequency
    """
    features_shape = tf.shape(features)
    n, v = features_shape[0], features_shape[1]

    f = tf.random_uniform([], 0, F, tf.int32)
    f0 = tf.random_uniform([], 0, v - f, tf.int32)
    mask = tf.concat(
        (
            tf.ones(shape = (n, v - f0 - f)),
            tf.zeros(shape = (n, f)),
            tf.ones(shape = (n, f0)),
        ),
        1,
    )
    masked = features * mask
    return tf.to_float(masked)


def tf_mask_time(features, T = 80):
    """
    Mask time using Tensorflow.
    
    Parameters
    ----------

    features : np.array
    T: size of mask for time
    """
    features_shape = tf.shape(features)
    n, v = features_shape[0], features_shape[1]
    t = tf.random_uniform([], 0, T, tf.int32)
    t0 = tf.random_uniform([], 0, n - T, tf.int32)
    mask = tf.concat(
        (
            tf.ones(shape = (n - t0 - t, v)),
            tf.zeros(shape = (t, v)),
            tf.ones(shape = (t0, v)),
        ),
        0,
    )
    masked = features * mask
    return tf.to_float(masked)
