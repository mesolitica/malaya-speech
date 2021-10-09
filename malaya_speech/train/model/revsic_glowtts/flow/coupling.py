from typing import Tuple

import tensorflow as tf


class AffineCoupling(tf.keras.Model):
    """Affine coupling layer.
    """

    def __init__(self, channels: int, hiddens: int, nonlinear: tf.keras.Model):
        """Initializer.
        Args:
            channels: size of the input/output channels.
            hiddens: size of the hidden channels.
            nonlinear: nonlinear transformer.
        """
        super().__init__()
        self.start = tf.keras.layers.Conv1D(hiddens, 1)
        self.end = tf.keras.layers.Conv1D(channels, 1, kernel_initializer='zeros')
        self.nonlinear = nonlinear

    def parameterize(self, inputs: tf.Tensor, mask: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        """Generate affine parameters.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
            mask: [tf.float32; [B, T]], binary sequence mask.
        Returns:
            bias: [tf.float23; [B, T, C]], bias tensor.
            logscale: [tf.float32; [B, T, C]], logscale tensor.
        """
        # [B, T, H]
        x = self.start(inputs) * mask[..., None]
        # [B, T, H]
        x = self.nonlinear(x, mask)
        # [B, T, C // 2], [B, T, C // 2]
        bias, logscale = tf.split(self.end(x), 2, axis=-1)
        return bias, logscale

    def call(self, inputs: tf.Tensor, mask: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        """Pass to affine coupling.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
            mask: [tf.float32; [B, T]], binary sequence mask.
        Returns:
            outputs: [tf.float32; [B, T, C]], transformed.
            dlogdet: [tf.float32; [B]], likelihood contribution.
        """
        # [B, T, C // 2], [B, T, C // 2]
        x0, x1 = tf.split(inputs, 2, axis=-1)
        # [B, T, C // 2], [B, T, C // 2]
        bias, logscale = self.parameterize(x0, mask)
        # [B, T, C // 2]
        x1 = bias + tf.exp(logscale) * x1
        # [B, T, C]
        outputs = tf.concat([x0, x1], axis=-1) * mask[..., None]
        # [B]
        dlogdet = tf.reduce_sum(logscale * mask[..., None], axis=[1, 2])
        return outputs, dlogdet

    def inverse(self, inputs: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """Inverse affine coupling.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
            mask: [tf.float32; [B, T]], binary sequence mask.
        Returns:
            [tf.float32; [B, T, C]], inversed.
        """
        # [B, T, C // 2], [B, T, C // 2]
        x0, x1 = tf.split(inputs, 2, axis=-1)
        # [B, T, C // 2], [B, T, C // 2]
        bias, logscale = self.parameterize(x0, mask)
        # [B, T, C // 2]
        x1 = (x1 - bias) * tf.exp(-logscale)
        # [B, T, C]
        return tf.concat([x0, x1], axis=-1) * mask[..., None]
