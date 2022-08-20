"""
MIT License

Copyright (c) 2021 YoungJoong Kim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

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
