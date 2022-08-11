import tensorflow as tf
from ..melgan.layer import WeightNormalization

LRELU_SLOPE = 0.1


class ResStack(tf.keras.layers.Layer):
    def __init__(self, channels=384, kernel_size=3, resstack_depth=3, hp=None, **kwargs):
        super().__init__(**kwargs)
        dilation = [2 * i + 1 for i in range(resstack_depth)]  # [1, 3, 5]
        self.convs1 = [
            WeightNormalization(
                tf.keras.layers.Conv1D(
                    channels, kernel_size, 1,
                    dilation_rate=dilation[i],
                    padding='SAME',
                    kernel_initializer=tf.random_normal_initializer(0, 0.01),
                )
            )
            for i in range(resstack_depth)
        ]

        self.convs2 = [
            WeightNormalization(
                tf.keras.layers.Conv1D(
                    channels, kernel_size, 1,
                    dilation_rate=1,
                    padding='SAME',
                    kernel_initializer=tf.random_normal_initializer(0, 0.01),
                )
            )
            for i in range(resstack_depth)
        ]

    def call(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = tf.keras.layers.LeakyReLU(alpha=LRELU_SLOPE)(x)
            xt = c1(xt)
            xt = tf.keras.layers.LeakyReLU(alpha=LRELU_SLOPE)(xt)
            xt = c2(xt)
            x = xt + x
        return x
