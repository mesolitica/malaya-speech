import tensorflow as tf


def weights_nonzero(labels):
    """Assign weight 1.0 to all labels except for padding (id=0)."""
    return to_float(tf.not_equal(labels, 0))


def ctc_symbol_loss(logits, targets, weights_fn = weights_nonzero):
    with tf.name_scope('ctc_loss', values = [logits, targets]):
        targets_mask = 1 - tf.to_int32(tf.equal(targets, 0))
        targets_lengths = tf.reduce_sum(targets_mask, axis = 1)

        sparse_targets = tf.keras.backend.ctc_label_dense_to_sparse(
            targets, targets_lengths
        )
        xent = tf.nn.ctc_loss(
            sparse_targets,
            logits,
            targets_lengths,
            time_major = False,
            preprocess_collapse_repeated = False,
            ctc_merge_repeated = False,
        )
        weights = weights_fn(targets)
        return tf.reduce_sum(xent), tf.reduce_sum(weights)
