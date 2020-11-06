import tensorflow as tf


def pad_second_dim(x, desired_size):
    padding = tf.tile(
        [[0]], tf.stack([tf.shape(x)[0], desired_size - tf.shape(x)[1]], 0)
    )
    return tf.concat([x, padding], 1)


def ctc_sequence_accuracy(
    logits, label, input_lengths, beam_width = 1, merge_repeated = True
):
    logits = tf.transpose(logits, [1, 0, 2])
    Y_seq_len = tf.count_nonzero(label, 1, dtype = tf.int32)

    decoded, log_prob = tf.nn.ctc_beam_search_decoder(
        logits,
        input_lengths,
        beam_width = beam_width,
        merge_repeated = merge_repeated,
    )

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
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


def ctc_sequence_accuracy_estimator(
    logits,
    label,
    input_lengths,
    beam_width = 1,
    merge_repeated = True,
    metrics_collections = None,
    updates_collections = None,
    name = None,
):
    accuracy = ctc_sequence_accuracy(
        logits = logits,
        label = label,
        input_lengths = input_lengths,
        beam_width = beam_width,
        merge_repeated = merge_repeated,
    )

    accuracy, update_accuracy_op = tf.metrics.mean(accuracy)

    if metrics_collections:
        tf.add_to_collections(metrics_collections, accuracy)

    if updates_collections:
        tf.add_to_collections(updates_collections, update_accuracy_op)

    return accuracy, update_accuracy_op


def weights_nonzero(labels):
    """Assign weight 1.0 to all labels except for padding (id=0)."""
    return to_float(tf.not_equal(labels, 0))


def from_tokens(raw, lookup):
    gathered = tf.gather(lookup, tf.cast(raw, tf.int32))
    joined = tf.regex_replace(
        tf.reduce_join(gathered, axis = 1), b'<EOS>.*', b''
    )
    cleaned = tf.regex_replace(joined, b'_', b' ')
    cleaned = tf.regex_replace(joined, b'<PAD>.*', b' ')
    tokens = tf.string_split(cleaned, ' ')
    return tokens


def from_characters(raw, lookup):
    """Convert ascii+2 encoded codes to string-tokens."""
    corrected = tf.bitcast(
        tf.clip_by_value(tf.subtract(raw, 2), 0, 255), tf.uint8
    )

    gathered = tf.gather(lookup, tf.cast(corrected, tf.int32))[:, :, 0]
    joined = tf.reduce_join(gathered, axis = 1)
    cleaned = tf.regex_replace(joined, b'\0', b'')
    tokens = tf.string_split(cleaned, ' ')
    return tokens


# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/metrics.py#L808
def word_error_rate(
    raw_predictions, labels, lookup = None, weights_fn = weights_nonzero
):
    """
    Calculate word error rate.
    Args:
        raw_predictions: The raw predictions.
        labels: The actual labels.
        lookup: A tf.constant mapping indices to output tokens.
        weights_fn: Weighting function.
    Returns:
        The word error rate.
    """

    if lookup is None:
        lookup = tf.constant([chr(i) for i in range(256)])
        convert_fn = from_characters
    else:
        convert_fn = from_tokens

    if weights_fn is not weights_nonzero:
        raise ValueError('Only weights_nonzero can be used for this metric.')

    with tf.variable_scope(
        'word_error_rate', values = [raw_predictions, labels]
    ):

        raw_predictions = tf.argmax(raw_predictions, axis = -1)
        labels = labels

        reference = convert_fn(labels, lookup)
        predictions = convert_fn(raw_predictions, lookup)

        distance = tf.reduce_sum(
            tf.edit_distance(predictions, reference, normalize = False)
        )
        reference_length = tf.cast(
            tf.size(reference.values, out_type = tf.int32), dtype = tf.float32
        )

        return distance / reference_length, reference_length


def word_error_rate_estimator(
    raw_predictions,
    labels,
    lookup = None,
    metrics_collections = None,
    updates_collections = None,
    name = None,
):
    wer, reference_length = word_error_rate(
        raw_predictions, labels, lookup = lookup
    )

    wer, update_wer_op = tf.metrics.mean(wer)

    if metrics_collections:
        tf.add_to_collections(metrics_collections, wer)

    if updates_collections:
        tf.add_to_collections(updates_collections, update_wer_op)

    return wer, update_wer_op
