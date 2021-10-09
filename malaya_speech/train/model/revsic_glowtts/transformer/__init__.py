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

from typing import List, Tuple

import tensorflow as tf

from .block import Block
from .pe import PositionalEncodings
from .prenet import Prenet
from ..config import Config


class Transformer(tf.keras.Model):
    """Transformer.
    """

    def __init__(self, config: Config):
        """Initializer.
        Args:
            config: transformer configuration.
                channels: int, size of the hidden channels.
                prenet_layers: int, the number of the prenet layers.
                prenet_kernel: int, size of the prenet kernels.
                prenet_dropout: float, dropout rate for prenet.
                block_ffn: int, size of the hidden channels for
                    feed-forward network.
                block_heads: int, the number of the attention heads.
                block_dropout: float, dropout rate for transformer blocks.
                block_num: int, the number of the attention blocks.
        """
        super(Transformer, self).__init__()
        self.prenet = Prenet(
            config.prenet_layers,
            config.channels,
            config.prenet_kernel,
            config.prenet_dropout)

        self.pe = PositionalEncodings(config.channels)

        self.blocks = [
            Block(
                config.channels,
                config.block_ffn,
                config.block_heads,
                config.block_dropout)
            for _ in range(config.block_num)]

    def call(self, inputs: tf.Tensor, mask: tf.Tensor) \
            -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """Transform the inputs.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
            mask: [tf.float32; [B, T]], mask tensor.
        Returns:
            x: [tf.float32; [B, T, C]], transformed tensor.
            attn: [tf.Tensor, [tf.float32; [B, K, T, T]]; N], attentions.
        """
        # [B, T, C]
        x = self.prenet(inputs, mask)
        # [B, T, C]
        x = x + self.pe(tf.shape(x)[1])[None]
        attn = []
        for block in self.blocks:
            # [B, T, C], [B, K, T, T]
            x, align = block(x, mask)
            # N x [B, K, T, T]
            attn.append(align)
        # [B, T, C], N x [B, K, T, T]
        return x, attn
