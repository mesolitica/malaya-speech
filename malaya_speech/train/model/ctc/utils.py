import tensorflow as tf


def calculate_input_length(inputs, logits, targets):
    seq_lens = tf.count_nonzero(tf.reduce_sum(inputs, -1), 1, dtype = tf.int32)
    targets_int32 = tf.cast(targets, tf.int32)
    filled = tf.fill(tf.shape(seq_lens), tf.reduce_max(targets_int32))
    seq_lens = tf.where(
        seq_lens < tf.reduce_max(targets_int32), filled, seq_lens
    )
    seq_lens = seq_lens // tf.cast(
        (tf.shape(inputs)[1] / tf.shape(logits)[1]), tf.int32
    )

    filled = tf.fill(tf.shape(seq_lens), tf.shape(model.logits)[1])
    seq_lens = tf.where(seq_lens > tf.shape(model.logits)[1], filled, seq_lens)
    return seq_lens


# https://github.com/tensorflow/models/blob/master/research/deep_speech/deep_speech.py#L42
def calculate_input_length_deep_speech(inputs, logits):
    input_length = tf.count_nonzero(
        tf.reduce_sum(inputs, -1), 1, dtype = tf.int32
    )
    max_time_steps = tf.shape(inputs)[1]
    ctc_time_steps = tf.shape(logits)[1]
    ctc_input_length = tf.cast(
        tf.multiply(input_length, ctc_time_steps), dtype = tf.float32
    )

    return tf.cast(
        tf.math.floordiv(
            ctc_input_length, tf.cast(max_time_steps, dtype = tf.float32)
        ),
        dtype = tf.int32,
    )
