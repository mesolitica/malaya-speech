import tensorflow as tf
from tensorflow.python.ops.init_ops import Initializer
from ..utils import shape_list

# check Table 4, https://arxiv.org/pdf/1911.13254.pdf
class ConvScaling(Initializer):
    def __init__(
        self, scale = 1.0, reference = 0.1, seed = None, dtype = tf.float32
    ):
        self.scale = scale
        self.reference = reference
        self.seed = seed
        self.dtype = dtype

    def __call__(self, shape, dtype = None, partition_info = None):
        stdv = 1.0 / (shape[0] * shape[1])
        w = tf.random.uniform(
            shape,
            minval = -stdv,
            maxval = stdv,
            dtype = self.dtype,
            seed = self.seed,
        )
        std = tf.math.reduce_std(w)
        scale = (std / self.reference) ** 0.5
        w = w / scale
        return w

    def get_config(self):
        return {
            'scale': self.scale,
            'seed': self.seed,
            'dtype': self.dtype.name,
        }


def glu(x, dims = 2, kernel_initializer = ConvScaling):
    def conv(x):
        return tf.layers.conv1d(
            x,
            x.shape[-1],
            1,
            padding = 'same',
            kernel_initializer = kernel_initializer,
        )

    splitted = tf.split(x, 2, dims)
    return conv(splitted[0]) * tf.nn.sigmoid(conv(splitted[1]))


def downsample(x, stride):
    return x[:, :, ::stride]


def center_trim(tensor, reference):
    reference = shape_list(reference)[-2]
    delta = shape_list(tensor)[-2] - reference
    if delta < 0:
        raise ValueError(
            f'tensor must be larger than reference. Delta is {delta}.'
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
    def __init__(
        self,
        filters,
        kernel_size,
        strides,
        activation,
        kernel_initializer = ConvScaling,
        **kwargs,
    ):
        super(Conv1DTranspose, self).__init__(
            name = 'Conv1DTranspose', **kwargs
        )
        self.conv = tf.keras.layers.Conv2DTranspose(
            filters,
            (kernel_size, 1),
            strides = (strides, 1),
            activation = activation,
            kernel_initializer = kernel_initializer,
        )

    def call(self, x):
        x = tf.expand_dims(x, 2)
        x = self.conv(x)
        return x[:, :, 0]
