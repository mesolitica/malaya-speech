import tensorflow as tf


class SpeechModel:
    def __init__(self, inputs, vocab_size = 256):
        self.convolution_count = 0

        outputs = tf.layers.conv1d(
            inputs,
            filters = 250,
            kernel_size = 48,
            stride = 2,
            padding = 'SAME',
            activation = tf.nn.relu,
        )
        for layer_idx in range(7):
            outputs = tf.layers.conv1d(
                outputs,
                filters = 250,
                kernel_size = 7,
                stride = 1,
                padding = 'SAME',
                activation = tf.nn.relu,
            )

        outputs = tf.layers.conv1d(
            outputs,
            filters = 250 * 8,
            kernel_size = 32,
            stride = 1,
            padding = 'SAME',
            activation = tf.nn.relu,
        )

        outputs = tf.layers.conv1d(
            outputs,
            filters = 250 * 8,
            kernel_size = 1,
            stride = 1,
            padding = 'SAME',
            activation = tf.nn.relu,
        )

        outputs = tf.layers.conv1d(
            outputs,
            filters = vocab_size,
            kernel_size = 1,
            stride = 1,
            padding = 'SAME',
            activation = None,
        )

        self.logits = outputs
