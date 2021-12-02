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
from malaya_speech.train.model.utils import shape_list


class Inv1x1Conv(tf.keras.Model):
    """Invertible 1x1 grouped convolution.
    """

    def __init__(self, groups):
        """Initializer.
        Args:
            groups: int, size of the convolution groups.
        """
        super(Inv1x1Conv, self).__init__()
        self.groups = groups
        # [groups, groups]
        weight, _ = tf.linalg.qr(tf.random.normal([groups, groups]))
        self.weight = tf.Variable(weight)

    def transform(self, inputs: tf.Tensor, mask: tf.Tensor, weight: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        """Convolve inputs.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
            mask: [tf.float32, [B, T]], sequence mask.
            weight: [tf.float32; [G, G]], convolutional weight.
        Returns:
            outputs: [tf.float32; [B, T, C]], convolved tensor.
            logdet: [tf.float32; [B]], log-determinant of conv2d derivation.
        """
        # [B, T, C // G, G]
        x = self.grouping(inputs)
        # [B, T, C // G, G]
        x = tf.nn.conv2d(x, weight[None, None], 1, padding='SAME')
        # []
        _, dlogdet = tf.linalg.slogdet(weight)
        # [B]
        dlogdet = dlogdet * tf.reduce_sum(mask, axis=-1) * \
            tf.cast(tf.shape(x)[2], tf.float32)
        # [B, T, C]
        outputs = self.recover(x)
        # [B, T, C], [B]
        return outputs, dlogdet

    def call(self, inputs: tf.Tensor, mask: tf.Tensor, g=None) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        """Forward 1x1 convolution.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
            mask: [tf.float32, [B, T]], sequence mask.
        Returns:
            outputs: [tf.float32; [B, T, C]], convolved tensor.
            logdet: [tf.float32; [B]], log-determinant of conv2d derivation.
        """
        return self.transform(inputs, mask, self.weight)

    def inverse(self, inputs: tf.Tensor, mask: tf.Tensor, g=None) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        """Inverse 1x1 convolution.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
            mask: [tf.float32, [B, T]], sequence mask.
        Returns:
            outputs: [tf.float32; [B, T, C]], convolved tensor.
        """
        outputs, _ = self.transform(inputs, mask, tf.linalg.inv(self.weight))
        return outputs

    def grouping(self, x: tf.Tensor) -> tf.Tensor:
        """Grouping tensor.
        Args:
            x: [tf.float32; [B, T, C]], input tensor.
        return:
            [tf.float32; [B, T, C // G, G]], grouped tensor.
        """
        # B, T, C
        bsize, timestep, channels = shape_list(x)
        # [B, T, 2, C // G, G // 2]
        x = tf.reshape(x, [bsize, timestep, 2, channels // self.groups, self.groups // 2])
        # [B, T, C // G, 2, G // 2]
        x = tf.transpose(x, [0, 1, 3, 2, 4])
        # [B, T, C // G, G]
        return tf.reshape(x, [bsize, timestep, channels // self.groups, self.groups])

    def recover(self, x: tf.Tensor) -> tf.Tensor:
        """Recover grouped tensor.
        Args:
            x: [tf.float32; [B, T, C // G, G]], grouped tensor.
        Returns:
            [tf.float32; [B, T, C]], recovered.
        """
        # B, T, C // G, G(=self.groups)
        bsize, timestep, splits, _ = shape_list(x)
        # [B, T, C // G, 2, G // 2]
        x = tf.reshape(x, [bsize, timestep, splits, 2, self.groups // 2])
        # [B, T, 2, C // G, G // 2]
        x = tf.transpose(x, [0, 1, 3, 2, 4])
        # [B, T, C]
        return tf.reshape(x, [bsize, timestep, splits * self.groups])
