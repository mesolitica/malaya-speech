import tensorflow as tf
from ..fastspeech.model import TFFastSpeechEncoder, TFTacotronPostnet
import numpy as np


def quantize_f0_tf(x, num_bins = 256):
    batch_size = tf.shape(x)[0]
    x = tf.reshape(x, (-1,))
    uv = x <= 0
    x = tf.where(uv, tf.zeros_like(x), x)
    x = tf.cast(tf.round((x / tf.reduce_max(x)) * (num_bins - 1)), tf.int32)
    x = x + 1
    x = tf.where(uv, tf.zeros_like(x), x)
    enc = tf.one_hot(x, num_bins + 1)
    return tf.reshape(enc, (batch_size, -1, num_bins + 1))


class InterpLnr(tf.keras.layers.Layer):
    def __init__(self, hparams, **kwargs):
        super().__init__(**kwargs)
        self.max_len_seq = hparams.max_len_seq
        self.max_len_pad = hparams.max_len_pad

        self.min_len_seg = hparams.min_len_seg
        self.max_len_seg = hparams.max_len_seg

        self.max_num_seg = self.max_len_seq // self.min_len_seg + 1

    def call(self, x, len_seq, training = True):

        if not training:
            return x

        batch_size = tf.shape(x)[0]
        dim = x.shape[2]
        indices = tf.tile(
            tf.expand_dims(tf.range(self.max_len_seg * 2), 0),
            (batch_size * self.max_num_seg, 1),
        )
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
        idx_scaled_org = idx_scaled_fl + tf.cast(offset, tf.float32)

        len_seq_rp = tf.repeat(len_seq, self.max_num_seg)
        idx_mask_org = idx_scaled_org < tf.cast(
            tf.expand_dims(len_seq_rp - 1, -1), tf.float32
        )

        idx_mask_final = tf.cast(idx_mask & idx_mask_org, tf.int32)

        counts = tf.reduce_sum(
            tf.reshape(
                tf.reduce_sum(idx_mask_final, axis = -1), (batch_size, -1)
            ),
            axis = -1,
        )

        index_1 = tf.repeat(tf.range(batch_size), counts)

        index_2_fl = tf.cast(
            tf.boolean_mask(idx_scaled_org, idx_mask_final), tf.int32
        )
        index_2_cl = index_2_fl + 1
        concatenated = tf.transpose(
            tf.concat(
                [
                    tf.expand_dims(index_1, axis = 0),
                    tf.expand_dims(index_2_fl, axis = 0),
                ],
                axis = 0,
            )
        )
        y_fl = tf.cast(tf.gather_nd(x, concatenated), tf.float32)

        concatenated = tf.transpose(
            tf.concat(
                [
                    tf.expand_dims(index_1, axis = 0),
                    tf.expand_dims(index_2_cl, axis = 0),
                ],
                axis = 0,
            )
        )
        y_cl = tf.cast(tf.gather_nd(x, concatenated), tf.float32)
        lambda_f = tf.expand_dims(tf.boolean_mask(lambda_, idx_mask_final), -1)

        y = (1 - lambda_f) * y_fl + lambda_f * y_cl
        sequences = tf.reshape(
            y[: (tf.shape(y)[0] // batch_size) * batch_size],
            (batch_size, -1, dim),
        )
        return tf.pad(
            sequences,
            ((0, 0), (0, self.max_len_pad - tf.shape(sequences)[1]), (0, 0)),
        )


class Encoder_7(tf.keras.layers.Layer):
    def __init__(self, hparams, **kwargs):
        super(Encoder_7, self).__init__(name = 'Encoder_7', **kwargs)


class Model(tf.keras.Model):
    def __init__(self, hparams, **kwargs):
        super(Model, self).__init__(name = 'speechsplit', **kwargs)
