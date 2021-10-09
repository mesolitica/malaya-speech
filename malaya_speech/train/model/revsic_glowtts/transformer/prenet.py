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
