import tensorflow as tf


def rnnt_loss(logits, labels, label_length, logit_length, blank = 0):
    try:
        from warprnnt_tensorflow import rnnt_loss as warp_rnnt_loss
    except:
        raise ModuleNotFoundError(
            'warprnnt_tensorflow not installed. Please install it by compile from https://github.com/usimarit/warp-transducer.git and try again.'
        )

    """
    Computes the RNNT loss between a sequence of activations and a ground truth labeling.
    Args:
        acts: A 4-D Tensor of floats.  The dimensions
                     should be (B, T, U+1, V), where B is the minibatch index,
                     T is the time index, U is the label sequence
                     length (+1 means blank label prepanded), 
                     and V indexes over activations for each 
                     symbol in the alphabet.
        labels: A 2-D Tensor of ints, a padded label sequences to make sure 
                     labels for the minibatch are same length.
        input_lengths: A 1-D Tensor of ints, the number of time steps
                       for each sequence in the minibatch.
        label_lengths: A 1-D Tensor of ints, the length of each label
                       for each example in the minibatch.
        blank_label: int, the label value/index that the RNNT
                     calculation should use as the blank label
    Returns:
        1-D float Tensor, the cost of each example in the minibatch
        (as negative log probabilities).
    * This class performs the softmax operation internally.
    * The label reserved for the blank symbol should be label 0.
    """
    if not tf.test.is_gpu_available():
        logits = tf.nn.log_softmax(logits)
    loss = warp_rnnt_loss(
        acts = tf.cast(logits, tf.float32),
        label_lengths = tf.cast(label_length, tf.int32),
        labels = tf.cast(labels, tf.int32),
        input_lengths = tf.cast(logit_length, tf.int32),
        blank_label = blank,
    )
    return loss
