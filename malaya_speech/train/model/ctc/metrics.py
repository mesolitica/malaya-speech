import tensorflow as tf


def weights_nonzero(labels):
    """Assign weight 1.0 to all labels except for padding (id=0)."""
    return to_float(tf.not_equal(labels, 0))


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

    def from_tokens(raw, lookup_):
        gathered = tf.gather(lookup_, tf.cast(raw, tf.int32))
        joined = tf.regex_replace(
            tf.reduce_join(gathered, axis = 1), b'<EOS>.*', b''
        )
        cleaned = tf.regex_replace(joined, b'_', b' ')
        tokens = tf.string_split(cleaned, ' ')
        return tokens

    def from_characters(raw, lookup_):
        """Convert ascii+2 encoded codes to string-tokens."""
        corrected = tf.bitcast(
            tf.clip_by_value(tf.subtract(raw, 2), 0, 255), tf.uint8
        )

        gathered = tf.gather(lookup_, tf.cast(corrected, tf.int32))[:, :, 0]
        joined = tf.reduce_join(gathered, axis = 1)
        cleaned = tf.regex_replace(joined, b'\0', b'')
        tokens = tf.string_split(cleaned, ' ')
        return tokens

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
    metrics_collections = None,
    updates_collections = None,
    name = None,
):
    wer, reference_length = word_error_rate(raw_predictions, labels)

    wer, update_wer_op = tf.metrics.mean(wer)

    if metrics_collections:
        tf.add_to_collections(metrics_collections, wer)

    if updates_collections:
        tf.add_to_collections(updates_collections, update_wer_op)

    return wer, update_wer_op
