import tensorflow as tf
import numpy as np
from itertools import permutations

EPS = 1e-8


def cal_abs_with_pit(source, estimate_source, source_lengths, C):

    estimate_source = tf.transpose(estimate_source, perm = [0, 2, 1, 3])

    mask = tf.cast(
        tf.sequence_mask(source_lengths, tf.reduce_max(source_lengths)),
        source.dtype,
    )
    mask = tf.expand_dims(mask, 1)
    mask = tf.expand_dims(mask, -1)
    estimate_source *= mask

    print(source, estimate_source)

    targets = tf.expand_dims(source, 1)
    est_targets = tf.expand_dims(estimate_source, 2)
    pw_loss = tf.abs(targets - est_targets)
    pair_wise_abs = tf.reduce_mean(pw_loss, axis = [3, 4])

    perms = tf.convert_to_tensor(np.array(list(permutations(range(C)))))
    perms = tf.cast(perms, tf.int32)
    index = tf.expand_dims(perms, 2)
    ones = tf.ones(tf.reduce_prod(tf.shape(index)))
    perms_one_hot = tf.zeros((tf.shape(perms)[0], tf.shape(perms)[1], C))

    indices = index
    tensor = perms_one_hot
    original_tensor = tensor
    indices = tf.reshape(indices, shape = [-1, tf.shape(indices)[-1]])
    indices_add = tf.expand_dims(
        tf.range(0, tf.shape(indices)[0], 1) * (tf.shape(tensor)[-1]), axis = -1
    )
    indices += indices_add
    tensor = tf.reshape(perms_one_hot, shape = [-1])
    indices = tf.reshape(indices, shape = [-1, 1])
    updates = tf.reshape(ones, shape = [-1])
    scatter = tf.tensor_scatter_nd_update(tensor, indices, updates)
    perms_one_hot = tf.reshape(
        scatter,
        shape = [
            tf.shape(original_tensor)[0],
            tf.shape(original_tensor)[1],
            -1,
        ],
    )

    abs_set = tf.einsum('bij,pij->bp', pair_wise_abs, perms_one_hot)
    min_abs = tf.reduce_min(abs_set, axis = 1, keepdims = True)
    min_abs /= C

    return min_abs


def calculate_loss(source, estimate_source, source_lengths, C):
    min_abs = cal_abs_with_pit(source, estimate_source, source_lengths, C)
    return tf.reduce_mean(min_abs)
