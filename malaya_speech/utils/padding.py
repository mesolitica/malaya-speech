import numpy as np


def padding_sequence_1d(seq, maxlen = None, padding: str = 'post', pad_int = 0):
    if padding not in ['post', 'pre']:
        raise ValueError('padding only supported [`post`, `pre`]')

    if not maxlen:
        maxlen = max([len(s) for s in seq])
    padded_seqs = []
    for s in seq:
        if padding == 'post':
            padded_seqs.append(s + [pad_int] * (maxlen - len(s)))
        if padding == 'pre':
            padded_seqs.append([pad_int] * (maxlen - len(s)) + s)
    return padded_seqs


def padding_sequence_nd(
    seq, maxlen = None, padding: str = 'post', pad_val = 0.0, dim: int = 1
):
    if padding not in ['post', 'pre']:
        raise ValueError('padding only supported [`post`, `pre`]')

    if not maxlen:
        maxlen = max([np.shape(s)[dim] for s in seq])

    padded_seqs = []
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
    return np.array(padded_seqs)
