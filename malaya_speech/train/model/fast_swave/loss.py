import tensorflow as tf
import numpy as np
from itertools import permutations
from tensorflow.python.ops import weights_broadcast_ops

EPS = 1e-8


def cal_abs_with_pit(source, estimate_source, source_lengths, C):

    estimate_source = tf.transpose(estimate_source, perm = [0, 2, 1, 3])

    mask = tf.cast(
        tf.sequence_mask(source_lengths, tf.reduce_max(source_lengths)),
        source.dtype,
    )
    mask = tf.expand_dims(mask, 1)
    mask = tf.expand_dims(mask, -1)
    # estimate_source *= mask

    targets = tf.expand_dims(source, 1)
    est_targets = tf.expand_dims(estimate_source, 2)
    pw_loss = tf.abs(targets - est_targets)
    # pair_wise_abs = tf.reduce_mean(pw_loss, axis = [3, 4])

    losses = pw_loss
    m = tf.expand_dims(mask, 1)
    weights = tf.cast(m, dtype = tf.float32)
    weighted_losses = tf.multiply(losses, weights)
    total_loss = tf.reduce_sum(weighted_losses, axis = [3, 4])
    present = tf.where(
        tf.equal(weights, 0.0), tf.zeros_like(weights), tf.ones_like(weights)
    )
    present = weights_broadcast_ops.broadcast_weights(present, losses)
    present = tf.reduce_sum(present, axis = [3, 4])
    pair_wise_abs = tf.div_no_nan(total_loss, present)

    v_perms = tf.constant(list(permutations(range(C))))
    perms_one_hot = tf.one_hot(v_perms, C)

    abs_set = tf.einsum('bij,pij->bp', pair_wise_abs, perms_one_hot)
    min_abs = tf.reduce_min(abs_set, axis = 1, keepdims = True)

    return min_abs


def calculate_loss(source, estimate_source, source_lengths, C):
    min_abs = cal_abs_with_pit(source, estimate_source, source_lengths, C)
    return tf.reduce_mean(min_abs)
