import tensorflow as tf
import numpy as np
from itertools import permutations
from ..utils import shape_list, log10

EPS = 1e-8


def cal_si_snr_with_pit(source, estimate_source, source_lengths, C):
    B, _, T = shape_list(source)
    mask = tf.cast(
        tf.sequence_mask(source_lengths, tf.reduce_max(source_lengths)),
        source.dtype,
    )
    mask = tf.expand_dims(mask, 1)
    estimate_source *= mask

    num_samples = tf.cast(tf.reshape(source_lengths, (-1, 1, 1)), tf.float32)
    mean_target = tf.reduce_sum(source, axis = 2, keepdims = True) / num_samples
    mean_estimate = (
        tf.reduce_sum(estimate_source, axis = 2, keepdims = True) / num_samples
    )
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate

    zero_mean_target *= mask
    zero_mean_estimate *= mask

    s_target = tf.expand_dims(zero_mean_target, 1)
    s_estimate = tf.expand_dims(zero_mean_estimate, 2)
    pair_wise_dot = tf.reduce_sum(
        s_estimate * s_target, axis = 3, keepdims = True
    )
    s_target_energy = (
        tf.reduce_sum(s_target ** 2, axis = 3, keepdims = True) + EPS
    )
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy
    e_noise = s_estimate - pair_wise_proj
    pair_wise_si_snr = tf.reduce_sum(pair_wise_proj ** 2, axis = 3) / (
        tf.reduce_sum(e_noise ** 2, axis = 3) + EPS
    )
    pair_wise_si_snr = 10.0 * log10(pair_wise_si_snr + EPS)
    pair_wise_si_snr = tf.transpose(pair_wise_si_snr, perm = [0, 2, 1])

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

    snr_set = tf.einsum('bij,pij->bp', pair_wise_si_snr, perms_one_hot)
    max_snr_idx = tf.argmax(snr_set, axis = 1)
    max_snr = tf.reduce_max(snr_set, axis = 1, keepdims = True)
    max_snr /= C

    return max_snr, perms, max_snr_idx, snr_set / C


def calculate_loss(source, estimate_source, source_lengths, C):
    max_snr, perms, max_snr_idx, snr_set = cal_si_snr_with_pit(
        source, estimate_source, source_lengths, C
    )
    loss = 0 - tf.reduce_mean(max_snr)
    return loss, max_snr, estimate_source
