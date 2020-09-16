import tensorflow as tf
import collections
import re


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
def calculate_input_length_v2(inputs, logits):
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


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match('^(.*):\\d+$', name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ':0'] = 1

    tf.logging.info('**** Trainable Variables ****')
    for var in tvars:
        init_string = ''
        if var.name in initialized_variable_names:
            init_string = ', *INIT_FROM_CKPT*'
        tf.logging.info(
            '  name = %s, shape = %s%s', var.name, var.shape, init_string
        )

    return (assignment_map, initialized_variable_names)
