from typing import Tuple

import numpy as np
import tensorflow as tf
from malaya_speech.train.model.utils import shape_list


class MultiHeadSelfAttn(tf.keras.Model):
    """Multihead Scaled-Dotproduct Self-attention block.
    """

    def __init__(self, channels: int, heads: int, dropout: float):
        """Initializer.
        Args:
            channels: output channels.
            heads: number of the heads.
            dropout: dropout rate.
        """
        super().__init__()
        self.heads = heads
        self.actual_channels = channels
        self.channels = channels // heads
        self.proj_q = tf.keras.layers.Dense(channels)
        self.proj_k = tf.keras.layers.Dense(channels)
        self.proj_v = tf.keras.layers.Dense(channels)

        self.dropout = tf.keras.layers.Dropout(dropout)
        self.proj_out = tf.keras.layers.Dense(channels)

    def call(self, inputs: tf.Tensor, mask: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        """Self-attend the inputs.
        Args:
            inputs: [tf.float32; [B, T, C]], input sequence.
            mask: [tf.float32; [B, T]], mask tensor.
        Returns:
            out: [tf.float32; [B, T, C]], attended sequence.
            attn: [tf.float32, [B, K, T, T]], dot-product attention.
        """
        # B, T, C
        bsize, timestep, _ = shape_list(inputs)
        # [B, T, K, Ck]
        query = tf.reshape(
            self.proj_q(inputs), [bsize, timestep, self.heads, -1])
        key = tf.reshape(
            self.proj_k(inputs), [bsize, timestep, self.heads, -1])
        value = tf.reshape(
            self.proj_v(inputs), [bsize, timestep, self.heads, -1])

        # [B, K, T, T]
        scores = tf.matmul(
            tf.transpose(query, [0, 2, 1, 3]),  # [B, K, T, Ck]
            tf.transpose(key, [0, 2, 3, 1]))    # [B, K, Ck, T]
        # scaling
        scores = scores / np.sqrt(self.channels)
        # masking
        scores = scores * mask[:, None, None] + \
            (1 - mask[:, None, None]) * tf.float32.min

        # [B, K, T, T]
        attn = tf.nn.softmax(scores, axis=-1)
        attn = self.dropout(attn)
        # [B, K, T, Ck]
        attend = tf.matmul(
            attn,
            tf.transpose(value, [0, 2, 1, 3]))  # [B, K, T, Ck]
        # [B, T, C]
        out = tf.reshape(
            tf.transpose(attend, [0, 2, 1, 3]),
            [bsize, timestep, self.actual_channels])
        # [B, T, Cout], [B, K, T, T]
        return self.proj_out(out) * mask[..., None], attn
