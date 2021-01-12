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


class Model:
    def __init__(
        self,
        inputs,
        dropout = 0.5,
        training = True,
        layers = 4,
        n_filters = [128, 384, 512, 512, 512, 512, 512, 512],
        n_filtersizes = [65, 33, 17, 9, 9, 9, 9, 9, 9],
        logging = True,
    ):
        with tf.name_scope('generator'):
            x = inputs
            L = layers
            downsampling_l = []
            for l, nf, fs in zip(range(L), n_filters, n_filtersizes):
                with tf.name_scope('downsc_conv%d' % l):
                    x = (
                        Conv1D(
                            filters = nf,
                            kernel_size = fs,
                            activation = None,
                            padding = 'same',
                            strides = 2,
                        )
                    )(x)
                    x = LeakyReLU(0.2)(x)

                    downsampling_l.append(x)

                    if logging:
                        print(x)

            with tf.name_scope('bottleneck_conv'):
                x = (
                    Conv1D(
                        filters = n_filters[-1],
                        kernel_size = n_filtersizes[-1],
                        activation = None,
                        padding = 'same',
                        strides = 2,
                    )
                )(x)
                x = Dropout(rate = dropout)(x, training = training)
                x = LeakyReLU(0.2)(x)

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
                            activation = None,
                            padding = 'same',
                        )
                    )(x)
                    x = Dropout(rate = dropout)(x, training = training)
                    x = Activation('relu')(x)
                    x = SubPixel1D(x, r = 2)
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
