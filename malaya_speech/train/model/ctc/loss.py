import tensorflow as tf


def to_float(x):
    return tf.cast(x, tf.float32)


def weights_nonzero(labels):
    return to_float(tf.not_equal(labels, 0))


def ctc_loss(logits, targets, input_lengths, weights_fn = weights_nonzero):
    with tf.name_scope('ctc_loss', values = [logits, targets]):
        targets_mask = 1 - tf.to_int32(tf.equal(targets, 0))
        targets_lengths = tf.reduce_sum(targets_mask, axis = 1)

        sparse_targets = tf.keras.backend.ctc_label_dense_to_sparse(
            targets, targets_lengths
        )
        xent = tf.nn.ctc_loss(
            sparse_targets, logits, input_lengths, time_major = False
        )
        weights = weights_fn(targets)
        return tf.reduce_mean(xent), tf.reduce_sum(xent), tf.reduce_sum(weights)


def keras_ctc_loss(
    logits, targets, input_lengths, weights_fn = weights_nonzero
):
    logits = tf.nn.softmax(logits)
    targets_mask = 1 - tf.to_int32(tf.equal(targets, 0))
    targets_lengths = tf.reduce_sum(targets_mask, axis = 1)
    xent = tf.keras.backend.ctc_batch_cost(
        targets,
        logits,
        tf.expand_dims(input_lengths, -1),
        tf.expand_dims(targets_lengths, -1),
    )
    weights = weights_fn(targets)
    return tf.reduce_mean(xent), tf.reduce_sum(xent), tf.reduce_sum(weights)
