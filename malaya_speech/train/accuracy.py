import tensorflow


def pad_second_dim(x, desired_size):
    padding = tf.tile(
        [[0]], tf.stack([tf.shape(x)[0], desired_size - tf.shape(x)[1]], 0)
    )
    return tf.concat([x, padding], 1)


def ctc_seq_accuracy(inputs, logits, label):
    seq_lens = tf.count_nonzero(tf.reduce_sum(inputs, -1), 1, dtype = tf.int32)
    Y_seq_len = tf.count_nonzero(label, 1, dtype = tf.int32)
    filled = tf.fill(tf.shape(seq_lens), tf.reduce_max(Y_seq_len))
    seq_lens = tf.where(seq_lens < tf.reduce_max(Y_seq_len), filled, seq_lens)

    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_lens)
    decoded = tf.to_int32(decoded[0])
    preds = tf.sparse.to_dense(decoded)
    preds = preds[:, : tf.reduce_max(Y_seq_len)]
    masks = tf.sequence_mask(
        Y_seq_len, tf.reduce_max(Y_seq_len), dtype = tf.float32
    )

    preds = pad_second_dim(preds, tf.reduce_max(Y_seq_len))
    y_t = tf.cast(preds, tf.int32)
    prediction = tf.boolean_mask(y_t, masks)
    mask_label = tf.boolean_mask(label, masks)
    correct_pred = tf.equal(prediction, mask_label)
    correct_index = tf.cast(correct_pred, tf.float32)
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))
