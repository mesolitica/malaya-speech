# https://github.com/kuleshov/audio-super-res/blob/master/src/models/audiotfilm.py

import tensorflow as tf
from tensorflow.keras.layers import (
    MaxPool1D,
    LeakyReLU,
    AveragePooling1D,
    BatchNormalization,
    Conv1D,
    Activation,
    Dropout,
    LSTM,
)


def SubPixel1D(I, r):
    with tf.name_scope('subpixel'):
        X = tf.transpose(I, [2, 1, 0])
        X = tf.batch_to_space_nd(X, [r], [[0, 0]])
        X = tf.transpose(X, [2, 1, 0])
        return X


DRATE = 2


class Model:
    def __init__(
        self,
        inputs,
        dropout = 0.5,
        training = True,
        layers = 4,
        n_filters = [128, 256, 512, 512, 512, 512, 512, 512],
        n_blocks = [128, 64, 32, 16, 8],
        n_filtersizes = [65, 33, 17, 9, 9, 9, 9, 9, 9],
        logging = True,
    ):
        with tf.name_scope('generator'):
            x = inputs
            L = layers
            downsampling_l = []

            def _make_normalizer(x_in, n_filters, n_block):
                x_shape = tf.shape(x_in)
                n_steps = x_shape[1] / n_block
                x_in_down = (MaxPool1D(pool_size = n_block, padding = 'valid'))(
                    x_in
                )
                x_rnn = LSTM(n_filters, return_sequences = True)(x_in_down)
                return x_rnn

            def _apply_normalizer(x_in, x_norm, n_filters, n_block):
                x_shape = tf.shape(x_in)
                n_steps = x_shape[1] / n_block
                x_in = tf.reshape(
                    x_in, shape = (-1, n_steps, n_block, n_filters)
                )
                x_norm = tf.reshape(x_norm, shape = (-1, n_steps, 1, n_filters))
                x_out = x_norm * x_in
                x_out = tf.reshape(x_out, shape = x_shape)
                return x_out

            for l, nf, fs in zip(range(L), n_filters, n_filtersizes):
                with tf.name_scope('downsc_conv%d' % l):

                    x = (
                        Conv1D(
                            filters = nf,
                            kernel_size = fs,
                            dilation_rate = DRATE,
                            activation = None,
                            padding = 'same',
                        )
                    )(x)
                    x = (MaxPool1D(pool_size = 2, padding = 'valid'))(x)
                    x = LeakyReLU(0.2)(x)

                    nb = 128 / (2 ** L)
                    x_norm = _make_normalizer(x, nf, int(nb))
                    x = _apply_normalizer(x, x_norm, nf, int(nb))

                    downsampling_l.append(x)

                    if logging:
                        print(x)

            with tf.name_scope('bottleneck_conv'):
                x = (
                    Conv1D(
                        filters = n_filters[-1],
                        kernel_size = n_filtersizes[-1],
                        dilation_rate = DRATE,
                        activation = None,
                        padding = 'same',
                    )
                )(x)
                x = (MaxPool1D(pool_size = 2, padding = 'valid'))(x)
                x = Dropout(rate = dropout)(x, training = training)
                x = LeakyReLU(0.2)(x)

                nb = 128 / (2 ** L)
                x_norm = _make_normalizer(x, n_filters[-1], int(nb))
                x = _apply_normalizer(x, x_norm, n_filters[-1], int(nb))

                if logging:
                    print(x)

            for l, nf, fs, l_in in reversed(
                list(zip(range(L), n_filters, n_filtersizes, downsampling_l))
            ):
                with tf.name_scope('upsc_conv%d' % l):
                    x = (
                        Conv1D(
                            filters = 2 * nf,
                            kernel_size = fs,
                            dilation_rate = DRATE,
                            activation = None,
                            padding = 'same',
                        )
                    )(x)
                    x = Dropout(rate = dropout)(x, training = training)
                    x = Activation('relu')(x)
                    x = SubPixel1D(x, r = 2)

                    x_norm = _make_normalizer(x, nf, int(nb))
                    x = _apply_normalizer(x, x_norm, nf, int(nb))

                    x = tf.concat([x, l_in], axis = -1)

                    if logging:
                        print(x)

            with tf.name_scope('lastconv'):
                x = (
                    Conv1D(
                        filters = 2,
                        kernel_size = 9,
                        activation = None,
                        padding = 'same',
                    )
                )(x)
                x = SubPixel1D(x, r = 2)

                if logging:
                    print(x)

            self.logits = x + inputs
