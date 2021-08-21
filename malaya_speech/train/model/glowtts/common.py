import numpy as np
import tensorflow as tf
from ..utils import shape_list


def maximum_path(value, mask, max_neg_val=-np.inf):
    """ Numpy-friendly version. It's about 4 times faster than torch version.
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    """
    value = value * mask

    device = value.device
    dtype = value.dtype
    mask = mask.astype(np.bool)

    b, t_x, t_y = value.shape
    direction = np.zeros(value.shape, dtype=np.int64)
    v = np.zeros((b, t_x), dtype=np.float32)
    x_range = np.arange(t_x, dtype=np.float32).reshape(1, -1)
    for j in range(t_y):
        v0 = np.pad(v, [[0, 0], [1, 0]], mode="constant", constant_values=max_neg_val)[:, :-1]
        v1 = v
        max_mask = (v1 >= v0)
        v_max = np.where(max_mask, v1, v0)
        direction[:, :, j] = max_mask

        index_mask = (x_range <= j)
        v = np.where(index_mask, v_max + value[:, :, j], max_neg_val)
    direction = np.where(mask, direction, 1)

    path = np.zeros(value.shape, dtype=np.float32)
    index = mask[:, :, 0].sum(1).astype(np.int64) - 1
    index_range = np.arange(b)
    for j in reversed(range(t_y)):
        path[index_range, index, j] = 1
        index = index + direction[index_range, index, j] - 1
    path = path * mask.astype(np.float32)
    return path


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = tf.reduce_max(length)
    x = tf.range(max_length, dtype=length.dtype)
    x = tf.expand_dims(x, 0)
    x = tf.tile(x, (tf.shape(length)[0], 1))
    ones = tf.ones_like(x)
    zeros = tf.zeros_like(x)
    return tf.where(x < tf.expand_dims(length, -1), ones, zeros)


def generate_path(duration, mask):
    """
    duration: [b, t_x]
    mask: [b, t_x, t_y]
    """
    b, t_x, t_y = shape_list(mask)

    cum_duration = tf.math.cumsum(duration, 1)
    path = tf.zeros((b, t_x, t_y), dtype=mask.dtype)
    cum_duration_flat = tf.reshape(cum_duration, (b * t_x))

    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = tf.reshape(path, (b, t_x, t_y))
    path = path - tf.pad(path, [[0, 0], [1, 0], [0, 0]])[:, :-1]
    path = path * mask
    return path
