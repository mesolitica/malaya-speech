import numpy as np
import tensorflow as tf
from typing import Tuple


def index_put(tensor, indices, value):
    value = tf.expand_dims(tf.expand_dims(value, axis = 0), axis = 0)
    tiled = tf.tile(value, (tf.shape(tensor)[0], tf.shape(tensor)[1], 1))
    indices = tf.tile(tf.expand_dims(indices, -1), (1, 1, tensor.shape[-1]))
    return tf.where(indices, tiled, tensor)


def index_put_constant(tensor, indices, value):
    tiled = tf.fill(tf.shape(tensor), value)
    return tf.where(indices, tiled, tensor)


def compute_mask_indices(
    shape,
    padding_mask,
    mask_prob: float,
    mask_length: int,
    mask_type: str = 'static',
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
):
    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)
    mask_type = mask_type.decode()

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None and not isinstance(padding_mask, bytes):
            sz = all_sz - padding_mask[i].sum()
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        if mask_type == 'static':
            lengths = np.full(num_mask, mask_length)
        elif mask_type == 'uniform':
            lengths = np.random.randint(
                mask_other, mask_length * 2 + 1, size = num_mask
            )
        elif mask_type == 'normal':
            lengths = np.random.normal(mask_length, mask_other, size = num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == 'poisson':
            lengths = np.random.poisson(mask_length, size = num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception('unknown mask selection ' + mask_type)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse = True):
                lens = np.fromiter(
                    (
                        e - s if e - s >= length + min_space else 0
                        for s, e in parts
                    ),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p = probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = np.random.choice(sz - min_len, num_mask, replace = False)

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace = False)
        mask[i, mask_idc] = True

    return mask
