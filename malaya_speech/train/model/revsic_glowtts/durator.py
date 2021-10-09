import tensorflow as tf


class DurationPredictor(tf.keras.Model):
    """Duration predictor.
    """

    def __init__(self, layers: int, channels: int, kernel: int, dropout: float):
        """Initializer.
        Args:
            layers: the number of the convolutional layers.
            channels: output channels.
            kernel: size of the kernel.
            dropout: dropout rate.
        """
        super().__init__()
        self.blocks = [self.conv_norm_relu(channels, kernel, dropout)
                       for _ in range(layers)]
        self.blocks.append(
            tf.keras.layers.Conv1D(1, kernel, 1, padding='SAME'))

    def call(self, inputs: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """Generate duration sequence.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
            mask: [tf.float32; [B, T]], binary tensor mask.
        Returns:
            [tf.float32; [B, T]], duration sequence.
        """
        # [B, T, C]
        x = inputs
        for layer in self.blocks:
            # [B, T, C]
            x = layer(x) * mask[..., None]
        # [B, T]
        return tf.squeeze(x, axis=-1)

    def conv_norm_relu(self, channels: int, kernel: int, dropout: float) \
            -> tf.keras.Sequential:
        """Generate sequential operation,
         Convolution - LayerNorm - ReLU - Dropout.

        Args:
            channels: output channels.
            kernel: size of the convolutional kernel.
            dropout: dropout rate.
        Returns:
            nonlinear operations.
        """
        return tf.keras.Sequential([
            tf.keras.layers.Conv1D(channels, kernel, 1, padding='SAME'),
            tf.keras.layers.LayerNormalization(axis=-1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(dropout)
        ])
