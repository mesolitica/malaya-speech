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

import tensorflow as tf


class Prenet(tf.keras.Model):
    """Transformer prenet.
    """

    def __init__(self, layers: int, channels: int, kernel: int, dropout: float):
        """Initializer.
        Args:
            layers: number of the prenet layers.
            channels: input channels.
            kernel: convolutional kernel size.
            dropout: dropout rate.
        """
        super().__init__()
        self.blocks = [
            self.conv_norm_relu(channels, kernel, dropout)
            for _ in range(layers)]
        self.residual = tf.keras.layers.Conv1D(
            channels, kernel, 1, padding='SAME',
            kernel_initializer='zeros',
            bias_initializer='zeros')

    def call(self, inputs: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """Pass to prenet.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
            mask: [tf.float32; [B, T]], sequential mask.
        Returns:
            [tf.float32; [B, T, C]], encoded sequence.
        """
        # [B, T, C]
        x = inputs
        for block in self.blocks:
            # [B, T, C]
            x = block(x) * mask[..., None]
        # [B, T, C]
        x = x + self.residual(inputs)
        return x * mask[..., None]

    def conv_norm_relu(self, channels: int, kernel: int, dropout: float) \
            -> tf.keras.Sequential:
        """Generate sequential operation,
         Convolution - LayerNorm - ReLU - Dropout.

        Args:
            channels: output channels.
            kernel: size of the convolutional kernel.
            dropout: dropout rate.
        Returns:
            tf.keras.Sequential, nonlinear operations.
        """
        return tf.keras.Sequential([
            tf.keras.layers.Conv1D(channels, kernel, 1, padding='SAME'),
            tf.keras.layers.LayerNormalization(axis=-1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(dropout)
        ])
