import numpy as np
import numpy.linalg as nl
import tensorflow as tf
import random

# https://github.com/NVIDIA/OpenSeq2Seq/blob/master/open_seq2seq/data/speech2text/speech_utils.py#L420
# https://github.com/Kyubyong/specAugment/blob/master/USER_DIR/speech_recognition.py
# https://github.com/KimJeongSun/SpecAugment_numpy_scipy
# https://espnet.github.io/espnet/_modules/espnet/transform/spec_augment.html


def warp_time_pil(features, max_time_warp = 80):
    from PIL import Image
    from PIL.Image import BICUBIC

    window = max_time_warp
    t = features.shape[0]
    if t - window <= window:
        return features
    center = random.randrange(window, t - window)
    warped = random.randrange(center - window, center + window) + 1

    left = Image.fromarray(features[:center]).resize(
        (features.shape[1], warped), BICUBIC
    )
    right = Image.fromarray(features[center:]).resize(
        (features.shape[1], t - warped), BICUBIC
    )
    return np.concatenate((left, right), 0)


def tf_warp_time(features, max_time_warp = 80):
    window = max_time_warp
    t = tf.shape(features)[0]

    def warp(features):
        center = tf.random.uniform(
            shape = [], minval = window, maxval = t - window, dtype = tf.int32
        )
        warped = (
            tf.random.uniform(
                shape = [],
                minval = center - window,
                maxval = center + window,
                dtype = tf.int32,
            )
            + 1
        )
        f = features[:center]
        im = f[tf.newaxis, :, :, tf.newaxis]
        left = tf.image.resize(
            im, (warped, features.shape[1]), method = 'bicubic'
        )
        f = features[center:]
        im = f[tf.newaxis, :, :, tf.newaxis]
        right = tf.image.resize(
            im, (t - warped, features.shape[1]), method = 'bicubic'
        )
        left = left[0, :, :, 0]
        right = right[0, :, :, 0]

        return tf.concat((left, right), 0)

    return tf.cond(
        t - window <= window, lambda: features, lambda: warp(features)
    )


def warp_time_interpolate(features, W = 40, T = 30, mt = 2):

    from scipy.spatial.distance import pdist, cdist, squareform
    from scipy import interpolate

    def makeT(cp):
        K = cp.shape[0]
        T = np.zeros((K + 3, K + 3))
        T[:K, 0] = 1
        T[:K, 1:3] = cp
        T[K, 3:] = 1
        T[K + 1 :, 3:] = cp.T
        R = squareform(pdist(cp, metric = 'euclidean'))
        R = R * R
        R[R == 0] = 1  # a trick to make R ln(R) 0
        R = R * np.log(R)
        np.fill_diagonal(R, 0)
        T[:K, 3:] = R
        return T

    def liftPts(p, cp):
        N, K = p.shape[0], cp.shape[0]
        pLift = np.zeros((N, K + 3))
        pLift[:, 0] = 1
        pLift[:, 1:3] = p
        R = cdist(p, cp, 'euclidean')
        R = R * R
        R[R == 0] = 1
        R = R * np.log(R)
        pLift[:, 3:] = R
        return pLift

    spec = features.T
    Nframe = spec.shape[1]
    Nbin = spec.shape[0]
    if Nframe < W * 2 + 1:
        W = int(Nframe / 4)
    if Nframe < T * 2 + 1:
        T = int(Nframe / mt)

    w = random.randint(-W, W)
    center = random.randint(W, Nframe - W)
    src = np.asarray(
        [
            [float(center), 1],
            [float(center), 0],
            [float(center), 2],
            [0, 0],
            [0, 1],
            [0, 2],
            [Nframe - 1, 0],
            [Nframe - 1, 1],
            [Nframe - 1, 2],
        ]
    )
    dst = np.asarray(
        [
            [float(center + w), 1],
            [float(center + w), 0],
            [float(center + w), 2],
            [0, 0],
            [0, 1],
            [0, 2],
            [Nframe - 1, 0],
            [Nframe - 1, 1],
            [Nframe - 1, 2],
        ]
    )

    xs, ys = src[:, 0], src[:, 1]
    cps = np.vstack([xs, ys]).T
    xt, yt = dst[:, 0], dst[:, 1]
    TT = makeT(cps)

    xtAug = np.concatenate([xt, np.zeros(3)])
    ytAug = np.concatenate([yt, np.zeros(3)])
    cx = nl.solve(TT, xtAug)
    cy = nl.solve(TT, ytAug)
    x = np.linspace(0, Nframe - 1, Nframe)
    y = np.linspace(1, 1, 1)
    x, y = np.meshgrid(x, y)

    xgs, ygs = x.flatten(), y.flatten()

    gps = np.vstack([xgs, ygs]).T

    pgLift = liftPts(gps, cps)
    xgt = np.dot(pgLift, cx.T)
    spec_warped = np.zeros_like(spec)
    for f_ind in range(Nbin):
        spec_tmp = spec[f_ind, :]
        func = interpolate.interp1d(xgt, spec_tmp, fill_value = 'extrapolate')
        xnew = np.linspace(0, Nframe - 1, Nframe)
        spec_warped[f_ind, :] = func(xnew)

    return spec_warped.T


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


def tf_mask_frequency(features, n_freq_mask = 2, F = 27):
    """
    Mask frequency using Tensorflow.

    Parameters
    ----------

    features : np.array
    F: size of mask for frequency
    """
    features_shape = tf.shape(features)
    n, v = features_shape[0], features_shape[1]

    for idx in range(n_freq_mask):

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
        features = masked
    return tf.to_float(masked)


def tf_mask_time(features, n_time_mask = 2, T = 80):
    """
    Mask time using Tensorflow.
    
    Parameters
    ----------

    features : np.array
    T: size of mask for time
    """
    features_shape = tf.shape(features)
    n, v = features_shape[0], features_shape[1]
    for idx in range(n_time_mask):
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
        features = masked
    return tf.to_float(masked)
