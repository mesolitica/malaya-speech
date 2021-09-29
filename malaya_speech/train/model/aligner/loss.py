import tensorflow as tf
from ..ctc import loss


def forwardsum_loss(attn_logprob, in_lens, out_lens, blank_logprob=0,
                    parallel_iterations=10, swap_memory=False):
    """
    attn_logprob: B x 1 x T1 x T2
    in_lens: B, batch of text length
    out_lens: B, batch of mel length
    """
    key_lens = in_lens
    query_lens = out_lens

    attn_logprob_padded = tf.pad(attn_logprob, ((0, 0), (0, 0), (0, 0), (1, 0)),
                                 constant_values=blank_logprob)
    batch = tf.constant(0, dtype=tf.int32)
    total_loss = tf.constant(0.0, dtype=tf.float32)
    batch_size = tf.shape(attn_logprob)[0]

    def condition(bid, total_loss):
        return tf.less(bid, batch_size)

    def body(bid, total_loss):
        target_seq = tf.expand_dims(tf.range(1, key_lens[bid] + 1), 0)
        curr_logprob = tf.transpose(attn_logprob_padded[bid], [1, 0, 2])[: query_lens[bid], :, : key_lens[bid] + 1]
        l, _, _ = loss.ctc_loss(
            tf.transpose(curr_logprob, [1, 0, 2]), target_seq, query_lens[bid: bid + 1]
        )
        total_loss += l
        return bid + 1, total_loss

    _, total_loss = tf.while_loop(
        condition,
        body,
        loop_vars=(batch, total_loss),
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory,
        back_prop=True,
    )
    return total_loss / tf.cast(batch_size, tf.float32)


def bin_loss(attn_hard, attn_soft):
    where = tf.boolean_mask(attn_soft, tf.equal(attn_hard, tf.ones_like(attn_hard)))
    log_sum = tf.math.log(tf.clip_by_value(where, 1e-12, tf.reduce_max(where)))
    log_sum = tf.reduce_sum(log_sum)
    return -log_sum / tf.reduce_sum(attn_hard)
