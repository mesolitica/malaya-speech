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

import numpy as np
import tensorflow as tf


class PositionalEncodings(tf.keras.Model):
    """Sinusoidal positional encoding generator.
    """

    def __init__(self, channels: int, presize: int = 128):
        """Initializer.
        Args:
            channels: size of the channels.
            presize: initial pe cache size.
        """
        super().__init__()
        self.channels = channels
        self.size = presize
        self.buffer = self.generate(presize)

    def call(self, size: int) -> tf.Tensor:
        """Return cached positional encodings.
        Args:
            size: length of the pe.
        Returns:
            [tf.float32; [T, C]], sinusoidal positional encodings.
        """
        if size <= self.size:
            return self.buffer[:size]
        # generate new cache
        self.buffer = self.generate(size)
        return self.buffer

    def generate(self, size: int) -> tf.Tensor:
        """Generate positional encodings.
        Args:
            size: length of the pe.
        Returns:
           [tf.float32; [T, C]], sinusoidal positional encodings.
        """
        # [tf.int32; [T]]
        pos = tf.range(size)
        # [tf.int32; [C//2]]
        i = tf.range(0, self.channels, 2)
        # [C//C], casting for float64
        denom = tf.exp(-np.log(10000) * tf.cast(i / self.channels, tf.float32))
        # [T, C//2]
        context = tf.cast(pos, tf.float32)[:, None] * denom[None]
        # [T, C//2, 1]
        context = context[..., None]
        # [T, C//2, 2]
        pe = tf.concat([tf.sin(context), tf.cos(context)], axis=-1)
        # [T, C]
        pe = tf.reshape(pe, [size, self.channels])
        return pe
