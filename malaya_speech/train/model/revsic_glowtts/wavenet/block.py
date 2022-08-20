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

from typing import Optional, Tuple

import tensorflow as tf
from malaya_speech.train.model.melgan.layer import WeightNormalization


class WaveNetBlock(tf.keras.Model):
    """Wavenet block.
    """

    def __init__(self,
                 channels: int,
                 kernel_size: int,
                 dilation: int,
                 cond: bool = False,
                 last: bool = False):
        """Initializer.
        Args:
            channels: channel size.
            kernel_size: kernel size of the dilated convolution.
            dilation: dilation rate.
            cond: whether use the auxiliary inputs or not.
            last: whether do not use additional projection
                for residual connection.
        """
        super().__init__()
        self.channels = channels
        self.cond = cond
        self.last = last

        self.dilated_conv = WeightNormalization(
            tf.keras.layers.Conv1D(
                channels * 2, kernel_size, strides=1, padding='same',
                dilation_rate=dilation),
            data_init=False)
        if cond:
            self.proj_aux = WeightNormalization(
                tf.keras.layers.Conv1d(channels * 2, 1))

        if not last:
            self.proj_res = WeightNormalization(
                tf.keras.layers.Conv1D(channels, 1))
        self.proj_skip = WeightNormalization(
            tf.keras.layers.Conv1D(channels, 1))

    def call(self,
             inputs: tf.Tensor,
             mask: tf.Tensor,
             aux: Optional[tf.Tensor] = None):
        """Pass wavenet block.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
            mask: [tf.float32; [B, T]], binary sequence mask.
            aux: [tf.float32; [B, T, H]], auxiliary inputs if provided.
        Returns:
            residual: [tf.float32; [B, T, C]], residually connected.
            skip: [tf.float32; [B, T, C]], outputs for skip connection.
        """
        # [B, T, Cx2]
        x = self.dilated_conv(inputs)
        if self.cond:
            assert aux is not None, 'auxiliary input is required'
            x = x + self.proj_aux(aux)
        # [B, T, C]
        context = tf.math.tanh(x[..., :self.channels])
        gate = tf.math.sigmoid(x[..., self.channels:])
        x = context * gate
        # [B, T, C]
        residual = (self.proj_res(x) + inputs) * mask[..., None] / (2 ** 0.5) \
            if not self.last else None
        skip = self.proj_skip(x) * mask[..., None]
        # [B, T, C], [B, T, C]
        return residual, skip
