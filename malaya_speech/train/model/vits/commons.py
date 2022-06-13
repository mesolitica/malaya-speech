import tensorflow as tf


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels
    in_act = input_a + input_b
    t_act = tf.tanh(in_act[:, :, :n_channels_int])
    s_act = tf.sigmoid(in_act[:, :, n_channels_int:])
    acts = t_act * s_act
    return acts


def sequence_mask(lengths, maxlen=None):
    if maxlen is None:
        maxlen = tf.reduce_max(lengths)
    return tf.cast(tf.range(maxlen)[None] < lengths[:, None], tf.float32)
