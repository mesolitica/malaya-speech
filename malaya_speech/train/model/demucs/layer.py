import tensorflow as tf
from ..utils import shape_list


def glu(x, dims = 2):
    def conv(x):
        return tf.layers.conv1d(x, x.shape[-1], 1, padding = 'same')

    splitted = tf.split(x, 2, dims)
    return conv(splitted[0]) * tf.nn.sigmoid(conv(splitted[1]))


def downsample(x, stride):
    return x[:, :, ::stride]


def center_trim(tensor, reference):
    reference = shape_list(reference)[-2]
    delta = shape_list(tensor)[-2] - reference
    if delta < 0:
        raise ValueError(
            'tensor must be larger than reference. ' f'Delta is {delta}.'
        )
    if delta:
        tensor = tensor[:, delta // 2 : -(delta - delta // 2)]
    return tensor


class BLSTM:
    def __init__(self, dim, num_layers = 2):
        self.dim = dim
        self.num_layers = num_layers

        self.lstm = tf.keras.Sequential()
        for _ in range(num_layers):
            self.lstm.add(
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(dim, return_sequences = True)
                )
            )
        self.dense = tf.keras.layers.Dense(dim)

    def __call__(self, x):
        x = self.lstm(x)
        return self.dense(x)


class Conv1DTranspose(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, activation, **kwargs):
        super(Conv1DTranspose, self).__init__(
            name = 'Conv1DTranspose', **kwargs
        )
        self.conv = tf.keras.layers.Conv2DTranspose(
            filters, kernel_size, strides = strides, activation = activation
        )

    def call(self, x):
        x = tf.expand_dims(x, 2)
        x = self.conv(x)
        return x[:, :, 0]
