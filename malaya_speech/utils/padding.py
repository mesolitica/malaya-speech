import numpy as np


def padding_sequence_1d(seq, maxlen = None, padding = 'post', pad_int = 0):
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
    seq, maxlen = None, padding = 'post', pad_val = 0.0, dim = 1
):
    if not maxlen:
        maxlen = max([np.shape(s)[dim] for s in seq])
    padded_seqs = []
