import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    LeakyReLU,
    Activation,
    Conv1D,
    ELU,
    Add,
)
from functools import partial
from tensorflow.compat.v1.keras.initializers import he_uniform


def _get_conv_activation_layer(params):
    """
    :param params:
    :returns: Required Activation function.
    """
    conv_activation = params.get('conv_activation')
    if conv_activation == 'ReLU':
        return ReLU()
    elif conv_activation == 'ELU':
        return ELU()
    return LeakyReLU(0.2)


class UpSamplingLayer:
    def __init__(self, channel_out, kernel_size = 5, stride = 1):
        self.seq = tf.keras.Sequential()
        self.seq.add(
            tf.keras.layers.Conv1D(
                channel_out,
                kernel_size = kernel_size,
                strides = stride,
                padding = 'SAME',
                dilation_rate = 1,
            )
        )
        self.seq.add(BatchNormalization(axis = -1))
        self.seq.add(LeakyReLU(0.2))

    def __call__(self, x, training = True):
        return self.seq(x, training = training)


class Model:
    def __init__(
        self,
        inputs,
        training = True,
        ksize = 5,
        n_layers = 12,
        channels_interval = 24,
        logging = True,
    ):
        conv_activation_layer = _get_conv_activation_layer({})
        kernel_initializer = he_uniform(seed = 50)

        conv1d_factory = partial(
            Conv1D,
            strides = (2),
            padding = 'same',
            kernel_initializer = kernel_initializer,
        )

        def resnet_block(input_tensor, filter_size):

            res = conv1d_factory(
                filter_size, (1), strides = (1), use_bias = False
            )(input_tensor)
            conv1 = conv1d_factory(filter_size, (5), strides = (1))(
                input_tensor
            )
            batch1 = BatchNormalization(axis = -1)(conv1, training = training)
            rel1 = conv_activation_layer(batch1)
            conv2 = conv1d_factory(filter_size, (5), strides = (1))(rel1)
            batch2 = BatchNormalization(axis = -1)(conv2, training = training)
            resconnection = Add()([res, batch2])
            rel2 = conv_activation_layer(resconnection)
            return rel2

        self.n_layers = n_layers
        self.channels_interval = channels_interval
        out_channels = [
            i * self.channels_interval for i in range(1, self.n_layers + 1)
        ]
        self.middle = tf.keras.Sequential()
        self.middle.add(
            tf.keras.layers.Conv1D(
                self.n_layers * self.channels_interval,
                kernel_size = 15,
                strides = 1,
                padding = 'SAME',
                dilation_rate = 1,
            )
        )
        self.middle.add(BatchNormalization(axis = -1))
        self.middle.add(LeakyReLU(0.2))

        decoder_out_channels_list = out_channels[::-1]

        self.decoder = []
        for i in range(self.n_layers):
            self.decoder.append(
                UpSamplingLayer(channel_out = decoder_out_channels_list[i])
            )
        self.out = tf.keras.Sequential()
        self.out.add(
            tf.keras.layers.Conv1D(
                1,
                kernel_size = 1,
                strides = 1,
                padding = 'SAME',
                dilation_rate = 1,
            )
        )
        self.out.add(Activation('tanh'))

        tmp = []
        o = inputs

        for i in range(self.n_layers):
            o = resnet_block(o, out_channels[i])
            tmp.append(o)
            o = o[:, ::2]
            if logging:
                print(o)

        o = self.middle(o, training = training)
        if logging:
            print(o)

        for i in range(self.n_layers):
            o = tf.image.resize(
                o, [tf.shape(o)[0], tf.shape(o)[1] * 2], method = 'nearest'
            )
            o = tf.concat([o, tmp[self.n_layers - i - 1]], axis = 2)
            o = self.decoder[i](o, training = training)
            if logging:
                print(o)

        if logging:
            print(o, inputs)
        o = tf.concat([o, inputs], axis = 2)
        o = self.out(o, training = training)
        self.logits = o
