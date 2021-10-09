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
import logging


class ActNorm(tf.keras.Model):
    """Activation normalization.
    """

    def __init__(self, channels: int):
        """Initializer.
        Args:
            channels: size of the input channels.
        """
        super().__init__()
        self.init = 0
        self.mean = tf.Variable(tf.zeros([channels]))
        self.logstd = tf.Variable(tf.zeros([channels]))

    def ddi(self, inputs: tf.Tensor, mask: tf.Tensor):
        """Data-dependent initialization.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
            mask: [tf.float32; [B, T]], binary sequence mask.
        """
        # []
        denom = tf.reduce_sum(mask)
        # [C]
        mean = tf.reduce_sum(inputs, axis=[0, 1]) / denom
        # [C]
        variance = tf.reduce_sum(tf.square(inputs), axis=[0, 1]) / denom - tf.square(mean)
        # [C]
        logstd = 0.5 * tf.math.log(tf.maximum(variance, 1e-5))
        # initialize
        self.mean.assign(mean)
        self.logstd.assign(logstd)
        self.init = 1

    def call(self, inputs: tf.Tensor, mask: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        """Normalize inputs with ddi.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
            mask: [tf.float32; [B, T]], binary sequence mask.
        Returns:
            outputs: [tf.float32; [B, T, C]], normalized.
            dlogdet: [tf.float32; [B]], likelihood contribution.
        """
        if self.init == 0:
            logging.info('initiate DDI')
            self.ddi(inputs, mask)
        # [B, T, C]
        outputs = (inputs - self.mean[None, None]) \
            * tf.exp(-self.logstd[None, None]) \
            * mask[..., None]
        # [B]
        dlogdet = tf.reduce_sum(-self.logstd) * tf.reduce_sum(mask, axis=1)
        return outputs, dlogdet

    def inverse(self, inputs: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """Denormalize inputs.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
            mask: [tf.float32; [B, T]], binary sequence mask.
        Returns:
            outputs: [tf.float32; [B, T, C]], denormalized.
        """
        # assert self.init == 1., "require ddi"
        # [B, T, C]
        x = inputs * tf.exp(self.logstd[None, None]) + self.mean[None, None]
        # [B, T, C]
        return x * mask[..., None]
