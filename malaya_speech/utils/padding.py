import numpy as np
import tensorflow as tf


def sequence_1d(
    seq, maxlen = None, padding: str = 'post', pad_int = 0, return_len = False
):
    """
    padding sequence of 1d to become 2d array.

    Parameters
    ----------
    seq: List[List[int]]
    maxlen: int, optional (default=None)
        If None, will calculate max length in the function.
    padding: str, optional (default='post')
        If `pre`, will add 0 on the starting side, else add 0 on the end side.
    pad_int, int, optional (default=0)
        padding value.

    Returns
    --------
    result: np.array
    """
    if padding not in ['post', 'pre']:
        raise ValueError('padding only supported [`post`, `pre`]')

    if not maxlen:
        maxlen = max([len(s) for s in seq])

    padded_seqs, length = [], []
    for s in seq:
        if isinstance(s, np.ndarray):
            s = s.tolist()
        if padding == 'post':
            padded_seqs.append(s + [pad_int] * (maxlen - len(s)))
        if padding == 'pre':
            padded_seqs.append([pad_int] * (maxlen - len(s)) + s)
        length.append(len(s))
    if return_len:
        return np.array(padded_seqs), length
    return np.array(padded_seqs)


def sequence_nd(
    seq,
    maxlen = None,
    padding: str = 'post',
    pad_val = 0.0,
    dim: int = 1,
    return_len = False,
):
    """
    padding sequence of nd to become (n+1)d array.

    Parameters
    ----------
    seq: list of nd array
    maxlen: int, optional (default=None)
        If None, will calculate max length in the function.
    padding: str, optional (default='post')
        If `pre`, will add 0 on the starting side, else add 0 on the end side.
    pad_val, float, optional (default=0.0)
        padding value.
    dim: int, optional (default=1)

    Returns
    --------
    result: np.array
    """
    if padding not in ['post', 'pre']:
        raise ValueError('padding only supported [`post`, `pre`]')

    if not maxlen:
        maxlen = max([np.shape(s)[dim] for s in seq])

    padded_seqs, length = [], []
    for s in seq:
        npad = [[0, 0] for _ in range(len(s.shape))]
        if padding == 'pre':
            padding = 0
        if padding == 'post':
            padding = 1
        npad[dim][padding] = maxlen - s.shape[dim]
        padded_seqs.append(
            np.pad(
                s,
                pad_width = npad,
                mode = 'constant',
                constant_values = pad_val,
            )
        )
        length.append(s.shape[dim])

    if return_len:
        return np.array(padded_seqs), length
    return np.array(padded_seqs)


def tf_sequence_nd(
    seq,
    maxlen = None,
    padding: str = 'post',
    pad_val = 0.0,
    dim: int = 1,
    return_len = False,
):
    """
    padding sequence of nd to become (n+1)d array.

    Parameters
    ----------
    seq: list of nd array
    maxlen: int, optional (default=None)
        If None, will calculate max length in the function.
    padding: str, optional (default='post')
        If `pre`, will add 0 on the starting side, else add 0 on the end side.
    pad_val, float, optional (default=0.0)
        padding value.
    dim: int, optional (default=1)

    Returns
    --------
    result: np.array
    """

    if not maxlen:
        maxlen = tf.reduce_max([tf.shape(seq[i])[0] for i in range(len(seq))])

    padded_seqs, length = [], []
    for i in range(len(seq)):
        s = seq[i]
        npad = [[0, 0] for _ in range(len(s.shape))]
        if padding == 'pre':
            padding = 0
        if padding == 'post':
            padding = 1
        npad[dim][padding] = maxlen - tf.shape(s)[dim]
        padded_seqs.append(
            tf.pad(
                s, paddings = npad, mode = 'CONSTANT', constant_values = pad_val
            )
        )
        length.append(tf.shape(s)[dim])

    if return_len:
        return tf.stack(padded_seqs), tf.stack(length)
    return tf.stack(padded_seqs)
