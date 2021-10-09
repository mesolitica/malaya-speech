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

from typing import Optional

import tensorflow as tf

from .block import WaveNetBlock


class WaveNet(tf.keras.Model):
    """WaveNet structure.
    """

    def __init__(self,
                 block_num_per_cycle: int,
                 cycle: int,
                 channels: int,
                 kernel_size: int,
                 dilation: int,
                 cond: bool = False):
        """Initializer.
        Args:
            block_num_per_cycle: the number of the blocks.
            cycle: the number of the cycles.
            channels: size of the hidden channels.
            kernel_size: size of the convolutional kernels.
            dilation: dilation rate for convolutional layers.
            cond: whether use auxiliary inputs or not.
        """
        super().__init__()
        self.block_num = block_num_per_cycle * cycle
        self.blocks = [
            # i = 0, 1, 2, ..., 0, 1, 2, ...
            WaveNetBlock(channels, kernel_size, dilation ** i, cond,
                         last=i == block_num_per_cycle - 1 and j == cycle - 1)
            for j in range(cycle)
            for i in range(block_num_per_cycle)]
        self.proj_skip = tf.keras.layers.Conv1D(channels, 1)

    def call(self,
             inputs: tf.Tensor,
             mask: tf.Tensor,
             aux: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Pass to WaveNet.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
            mask: [tf.float32; [B, T]], binary mask.
            aux: [tf.float32; [B, T, H]], auxiliary inputs if provided.
        Returns:
            [tf.float32; [B, T, C]], transformed.
        """
        context = []
        # [B, T, C]
        x = inputs
        for block in self.blocks:
            # [B, T, C], [B, T, C]
            x, skip = block(x, mask, aux)
            context.append(skip)
        # [B, T, C], variance scaling
        context = tf.reduce_sum(context, axis=0) / (self.block_num ** 0.5)
        # [B, T, C]
        return self.proj_skip(context) * mask[..., None]
