import tensorflow as tf


def glu(x, dims = 1):
    def conv(x):
        return tf.layers.conv1d(x, x.shape[-1], 1, padding = 'same')

    splitted = tf.split(x, 2, dims)
    return conv(splitted[0]) * tf.nn.sigmoid(conv(splitted[1]))
