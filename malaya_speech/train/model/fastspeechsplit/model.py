import tensorflow as tf


class InterpLnr(tf.keras.layers.Layer):
    def __init__(
        self,
        min_len_seg = 19,
        max_len_seg = 32,
        min_len_seq = 64,
        max_len_seq = 128,
        max_len_pad = 192,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_len_seq = max_len_seq
        self.max_len_pad = max_len_pad

        self.min_len_seg = min_len_seg
        self.max_len_seg = max_len_seg

        self.max_num_seg = self.max_len_seq // self.min_len_seg + 1

    def call(self, x, len_seq, training = True):
        if not training:
            return x

        batch_size = tf.shape(x)[0]
        x = tf.expand_dims(tf.range(self.max_len_seg * 2), 0)
        indices = tf.tile(x, (batch_size * self.max_num_seg, 1))
        scales = (
            tf.random.uniform(shape = (batch_size * self.max_num_seg,)) + 0.5
        )
        idx_scaled = tf.cast(indices, tf.float32) / tf.expand_dims(scales, -1)
        idx_scaled_fl = tf.math.floor(idx_scaled)
        lambda_ = idx_scaled - idx_scaled_fl
        len_seg = tf.random.uniform(
            (batch_size * self.max_num_seg, 1),
            minval = self.min_len_seg,
            maxval = self.max_len_seg,
            dtype = tf.int32,
        )
        idx_mask = idx_scaled_fl < (tf.cast(len_seg, tf.float32) - 1)
        offset = tf.math.cumsum(
            tf.reshape(len_seg, (batch_size, -1)), axis = -1
        )
        offset = tf.reshape(tf.pad(offset[:, :-1], ((0, 0), (1, 0))), (-1, 1))
        idx_scaled_org = idx_scaled_fl + offset

        len_seq_rp = tf.repeat(len_seq, self.max_num_seg)
        idx_mask_org = idx_scaled_org < tf.expand_dims(len_seq_rp - 1, -1)

        idx_mask_final = idx_mask & idx_mask_org

        counts = tf.reduce_sum(
            tf.reshape(
                tf.reduce_sum(idx_mask_final, axis = -1), (batch_size, 1)
            ),
            axis = -1,
        )

        index_1 = tf.repeat(tf.range(batch_size), counts)
        index_2_fl = idx_scaled_org[idx_mask_final]
        index_2_cl = index_2_fl + 1

        y_fl = x[index_1, index_2_fl, :]
        y_cl = x[index_1, index_2_cl, :]
        lambda_f = tf.expand_dims(lambda_[idx_mask_final], -1)

        y = (1 - lambda_f) * y_fl + lambda_f * y_cl

        sequences = tf.split(y, counts, axis = 0)

        # seq_padded = self.pad_sequences(sequences)
        # return seq_padded
        # tf.pad(sequences, )
        # return seq_padded
