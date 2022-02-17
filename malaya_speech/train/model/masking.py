import tensorflow as tf
from .utils import shape_list

# https://github.com/huggingface/transformers/blob/master/src/transformers/models/wav2vec2/modeling_tf_wav2vec2.py


def _sample_without_replacement(distribution, num_samples):
    """
    Categorical sampling without replacement is currently not implemented. The gumbel-max trick will do for now - see
    https://github.com/tensorflow/tensorflow/issues/9260 for more info
    """
    z = -tf.math.log(tf.random.uniform(shape_list(distribution), 0, 1))
    _, indices = tf.nn.top_k(distribution + z, num_samples)
    return indices


def _scatter_values_on_batch_indices(values, batch_indices, output_shape):
    """
    Scatter function as in PyTorch with indices in format (batch_dim, indixes)
    """
    indices_shape = shape_list(batch_indices)
    # broadcast batch dim to indices_shape
    broad_casted_batch_dims = tf.reshape(
        tf.broadcast_to(tf.expand_dims(tf.range(indices_shape[0]), axis=-1), indices_shape), [1, -1]
    )
    # transform batch_indices to pair_indices
    pair_indices = tf.transpose(tf.concat([broad_casted_batch_dims, tf.reshape(batch_indices, [1, -1])], 0))
    # scatter values to pair indices
    return tf.scatter_nd(pair_indices, tf.reshape(values, [-1]), output_shape)


def compute_mask_indices(
    shape,
    mask_prob: float,
    mask_length: int,
    min_masks: int = 0,
):
    """
    Computes random mask spans for a given shape
    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        attention_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob:
            probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_length: size of the mask
        min_masks: minimum number of masked spans
    Adapted from [fairseq's
    data_utils.py](https://github.com/pytorch/fairseq/blob/e0788f7007a8473a76db573985031f3c94201e79/fairseq/data/data_utils.py#L376).
    """
    batch_size, sequence_length = shape

    # compute number of masked spans in batch
    num_masked_spans = mask_prob * tf.cast(sequence_length, tf.float32) / mask_length + tf.random.uniform((1,))[0]
    num_masked_spans = tf.cast(num_masked_spans, tf.int32)
    num_masked_spans = tf.maximum(num_masked_spans, min_masks)

    # make sure num masked indices <= sequence_length
    num_masked_spans = tf.cond(tf.math.greater(num_masked_spans * mask_length, sequence_length),
                               lambda: sequence_length // mask_length,
                               lambda: num_masked_spans)

    # SpecAugment mask to fill
    spec_aug_mask = tf.zeros((batch_size, sequence_length), dtype=tf.int32)

    # uniform distribution to sample from, make sure that offset samples are < sequence_length
    uniform_dist = tf.ones((batch_size, sequence_length - (mask_length - 1)))

    # get random indices to mask
    spec_aug_mask_idxs = _sample_without_replacement(uniform_dist, num_masked_spans)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = tf.expand_dims(spec_aug_mask_idxs, -1)
    spec_aug_mask_idxs = tf.tile(spec_aug_mask_idxs, (1, 1, mask_length))
    spec_aug_mask_idxs = tf.reshape(spec_aug_mask_idxs, (batch_size, num_masked_spans * mask_length))

    offsets = tf.range(mask_length)[tf.newaxis, tf.newaxis, :]
    offsets = tf.tile(offsets, (batch_size, num_masked_spans, 1))
    offsets = tf.reshape(offsets, (batch_size, num_masked_spans * mask_length))

    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # scatter indices to mask
    spec_aug_mask = _scatter_values_on_batch_indices(
        tf.ones_like(spec_aug_mask_idxs), spec_aug_mask_idxs, tf.shape(spec_aug_mask)
    )

    return spec_aug_mask
