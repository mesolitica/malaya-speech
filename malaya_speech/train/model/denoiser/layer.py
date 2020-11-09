import math
import tensorflow as tf


class BLSTM:
    def __init__(self, dim, num_layers = 2, bi = True):
        self.dim = dim
        self.num_layers = num_layers

        self.lstm = tf.keras.Sequential()
        for _ in range(num_layers):
            if bi:
                layer = tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(dim, return_sequences = True)
                )
            else:
                layer = tf.keras.layers.LSTM(dim, return_sequences = True)
            self.lstm.add(layer)
        self.dense = tf.keras.layers.Dense(dim)

    def __call__(self, x):
        x = self.lstm(x)
        return self.dense(x)


def sinc(t):
    return tf.where(tf.equal(t, 0.0), tf.fill(t.shape, 1.0), tf.math.sin(t) / t)


def kernel_upsample2(zeros = 56):
    win = tf.signal.hann_window(4 * zeros + 1, periodic = False)
    winodd = win[1::2]
    t = tf.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t *= math.pi
    kernel = sinc(t) * winodd
    return tf.reshape(kernel, (-1, 1, 1))


def upsample2(x, zeros = 56):
    kernel = kernel_upsample2(zeros)

    s = tf.reshape(x, (-1, tf.shape(x)[1], 1))
    s = tf.pad(s, [[0, 0], [zeros, zeros], [0, 0]])
    convd = tf.nn.conv1d(s, kernel, padding = 'VALID')
    convd = convd[:, 1:]
    y = tf.concat([x, convd], axis = 1)
    return y


def kernel_downsample2(zeros = 56):
    win = tf.signal.hann_window(4 * zeros + 1, periodic = False)
    winodd = win[1::2]
    t = tf.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t *= math.pi
    kernel = sinc(t) * winodd
    return tf.reshape(kernel, (-1, 1, 1))


def downsample2(x, zeros = 56):
    x = tf.cond(
        tf.math.not_equal(tf.mod(tf.shape(x)[1], 2), 0),
        lambda: tf.pad(x, [[0, 0], [0, 1], [0, 0]]),
        lambda: x,
    )
    xeven = x[:, ::2]
    xodd = x[:, 1::2]
    kernel = kernel_downsample2(zeros)
    s = tf.reshape(xodd, (-1, tf.shape(xodd)[1], 1))
    s = tf.pad(s, [[0, 0], [zeros, zeros], [0, 0]])
    convd = tf.nn.conv1d(s, kernel, padding = 'VALID')
    convd = xeven + convd[:, :-1]
    return tf.multiply(convd, 0.5)
