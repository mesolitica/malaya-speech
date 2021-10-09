from typing import Tuple

import tensorflow as tf

from .attention import MultiHeadSelfAttn


class Block(tf.keras.Model):
    """Transformer block.
    """

    def __init__(self, channels: int, ffn: int, heads: int, dropout: float):
        """Initializer.
        Args:
            channels: input channels,
            ffn: size of the ffn channels.
            heads: the number of the attention heads.
            dropout: dropout rate.
        """
        super().__init__()
        # attention block.
        self.attn = MultiHeadSelfAttn(channels, heads, dropout)
        self.norm1 = tf.keras.Sequential([
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.LayerNormalization(axis=-1)])
        # ffn block
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ffn),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(channels)])
        self.norm2 = tf.keras.Sequential([
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.LayerNormalization(axis=-1)])

    def call(self, inputs: tf.Tensor, mask: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        """Transform the inputs.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
            mask: [tf.float32; [B, T]], sequence mask.
        Returns:
            y: [tf.float32; [B, T, C]], transformed inputs.
            attn: [tf.float32; [B, K, T, T]], attention weights.
        """
        # [B, T, C], [B, K, T, T]
        x, attn = self.attn(inputs, mask)
        # [B, T, C], assume inputs tensor is already masked
        x = self.norm1(x + inputs) * mask[..., None]
        # [B, T, C]
        y = self.ffn(x) * mask[..., None]
        # [B, T, C]
        y = self.norm2(y + x) * mask[..., None]
        return y, attn
