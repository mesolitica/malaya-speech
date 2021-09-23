import tensorflow as tf
import numpy as np

mask_value = -1e9


def binarize_attention(attn, in_len, out_len):
    b_size = attn.shape[0]
    attn_cpu = attn
    attn_out = np.zeros_like(attn)
    for ind in range(b_size):
        hard_attn = mas(attn_cpu[ind, 0, : out_len[ind], : in_len[ind]])
        attn_out[ind, 0, : out_len[ind], : in_len[ind]] = hard_attn
    return attn_out


def mas(attn_map, width=1):
    # assumes mel x text
    opt = np.zeros_like(attn_map)
    attn_map = np.log(attn_map)
    attn_map[0, 1:] = -np.inf
    log_p = np.zeros_like(attn_map)
    log_p[0, :] = attn_map[0, :]
    prev_ind = np.zeros_like(attn_map, dtype=np.int64)
    for i in range(1, attn_map.shape[0]):
        for j in range(attn_map.shape[1]):
            prev_j = np.arange(max(0, j - width), j + 1)
            prev_log = np.array([log_p[i - 1, prev_idx] for prev_idx in prev_j])

            ind = np.argmax(prev_log)
            log_p[i, j] = attn_map[i, j] + prev_log[ind]
            prev_ind[i, j] = prev_j[ind]

    # now backtrack
    curr_text_idx = attn_map.shape[1] - 1
    for i in range(attn_map.shape[0] - 1, -1, -1):
        opt[i, curr_text_idx] = 1
        curr_text_idx = prev_ind[i, curr_text_idx]
    opt[0, curr_text_idx] = 1

    assert opt.sum(0).all()
    assert opt.sum(1).all()

    return opt


def get_initializer(initializer_range=0.02):
    """Creates a `tf.initializers.truncated_normal` with the given range.
    Args:
        initializer_range: float, initializer range for stddev.
    Returns:
        TruncatedNormal initializer with stddev = `initializer_range`.
    """
    return tf.keras.initializers.TruncatedNormal(
        mean=0.0, stddev=initializer_range
    )


class ConvNorm(tf.keras.layers.Layer):
    def __init__(
        self,
        out_channels,
        kernel_size=1,
        stride=1,
        padding='SAME',
        dilation=1,
        bias=True,
        activation=None,
        **kwargs,
    ):
        super(ConvNorm, self).__init__(name='ConvNorm', **kwargs)
        self.conv = tf.keras.layers.Conv1D(
            out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            dilation_rate=dilation,
            use_bias=bias,
            activation=activation,
        )

    def call(self, x):
        return self.conv(x)


class AlignmentEncoder(tf.keras.Model):
    def __init__(
        self, vocab_size, vocab_embedding,
        n_mel_channels=80, n_text_channels=512, n_att_channels=80, temperature=0.0005,
        use_position_embedding=False, max_position_embeddings=2048, **kwargs
    ):
        super(AlignmentEncoder, self).__init__(name='AlignmentEncoder', **kwargs)
        self.temperature = temperature

        self.key_proj = [ConvNorm(n_text_channels * 2, kernel_size=3, bias=True, activation=tf.nn.relu),
                         ConvNorm(n_att_channels, kernel_size=1, bias=True)]
        self.key_proj = tf.keras.Sequential(self.key_proj)

        self.query_proj = [ConvNorm(n_mel_channels * 2, kernel_size=3, bias=True, activation=tf.nn.relu),
                           ConvNorm(n_mel_channels, kernel_size=1, bias=True, activation=tf.nn.relu),
                           ConvNorm(n_att_channels, kernel_size=1, bias=True)]
        self.query_proj = tf.keras.Sequential(self.query_proj)
        self.embeddings = tf.get_variable(
            'AlignmentEncoder/embeddings',
            [vocab_size, vocab_embedding],
            dtype=tf.float32,
            initializer=get_initializer(),
        )
        self.hidden_size = vocab_embedding
        self.max_position_embeddings = max_position_embeddings
        self.use_position_embedding = use_position_embedding
        if self.use_position_embedding:
            self.position_embeddings = tf.convert_to_tensor(
                self._sincos_embedding()
            )

    def _sincos_embedding(self):
        position_enc = np.array(
            [
                [
                    pos
                    / np.power(10000, 2.0 * (i // 2) / self.hidden_size)
                    for i in range(self.hidden_size)
                ]
                for pos in range(self.max_position_embeddings + 1)
            ]
        )

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])

        # pad embedding.
        position_enc[0] = 0.0

        return position_enc

    def call(self, queries, keys, mask=None, attn_prior=None, training=True, **kwargs):
        """
        Args:
        queries (torch.tensor): B x T1 x C1 tensor (probably going to be mel data).
        keys (torch.tensor): B x T2 x C2 tensor (text data).
        mask (torch.tensor): uint8 binary mask for variable length entries, B x T2 x 1 (should be in the T2 domain).
        attn_prior (torch.tensor): prior for attention matrix, B x T1 x T2
        Output:
            attn (torch.tensor): B x 1 x T1 x T2 attention mask. Final dim T2 should sum to 1.
            attn_logprob (torch.tensor): B x 1 x T1 x T2 log-prob attention mask.
        """
        keys = tf.nn.embedding_lookup(self.embeddings, keys)
        if self.use_position_embedding:
            input_shape = tf.shape(keys)
            seq_length = input_shape[1]
            inputs = tf.range(1, seq_length + 1, dtype=tf.int32)[
                tf.newaxis, :
            ]
            position_embeddings = tf.gather(self.position_embeddings, inputs)
            keys = keys + tf.cast(
                position_embeddings, keys.dtype
            )

        keys_enc = self.key_proj(keys, training=training)
        queries_enc = self.query_proj(queries, training=training)
        keys_enc = tf.transpose(keys_enc, [0, 2, 1])
        queries_enc = tf.transpose(queries_enc, [0, 2, 1])

        attn = (tf.expand_dims(queries_enc, -1) - tf.expand_dims(keys_enc, 2)) ** 2
        attn = -self.temperature * tf.reduce_sum(attn, 1, keepdims=True)

        if attn_prior is not None:
            attn = tf.nn.log_softmax(attn, 3) + tf.math.log(tf.expand_dims(attn_prior, 1) + 1e-8)

        attn_logprob = tf.identity(attn)

        if mask is not None:
            mask = tf.expand_dims(tf.transpose(mask, [0, 2, 1]), 2)
            mask = tf.tile(mask, (1, 1, tf.shape(attn)[2], 1))
            fill = tf.fill(tf.shape(mask), mask_value)
            attn = tf.where(tf.cast(mask, tf.bool), attn, fill)

        return tf.nn.softmax(attn, 3), attn_logprob

    def get_hard_attention(self, attn_soft, in_lens, out_lens):
        """
        attn_logprob: B x 1 x T1 x T2
        in_lens: B, batch of text length
        out_lens: B, batch of mel length
        """
        attn_hard = tf.compat.v1.numpy_function(
            binarize_attention,
            [
                attn_soft,
                in_lens,
                out_lens,
            ],
            tf.float32,
        )
        attn_hard.set_shape(attn_soft.shape)
        return attn_hard
