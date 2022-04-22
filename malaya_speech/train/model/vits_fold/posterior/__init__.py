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

from ..wavenet import WaveNet


class PosteriorEncoder(tf.keras.Model):

    def __init__(self, config):
        super().__init__()
        self.pre = tf.keras.layers.Conv1D(config.channels, 1)
        self.enc = WaveNet(
            8,
            config.wavenet_cycle,
            config.channels,
            config.wavenet_kernel_size,
            config.wavenet_dilation)
        self.proj = tf.keras.layers.Conv1D(config.neck * 2, 1)
        self.interchannels = config.neck

    def call(self, inputs: tf.Tensor, mask: tf.Tensor):
        """
        Compute latent from inputs.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
            mask: [tf.float32; [B, T]], binary sequence mask.
        """
        x = self.pre(inputs) * mask[..., None]
        x = self.enc(x, mask)
        stats = self.proj(x) * mask[..., None]
        m, logs = tf.split(stats, 2, axis=-1)
        z = (m + tf.random.normal(tf.shape(m)) * tf.exp(logs)) * mask[..., None]
        return z, m, logs
