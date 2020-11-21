import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    ELU,
    LeakyReLU,
    Multiply,
    ReLU,
    Softmax,
    Add,
    MaxPooling2D,
)
from tensorflow.compat.v1.keras.initializers import he_uniform
from functools import partial


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


def _get_deconv_activation_layer(params):
    """
    :param params:
    :returns: Required Activation function.
    """
    deconv_activation = params.get('deconv_activation')
    if deconv_activation == 'LeakyReLU':
        return LeakyReLU(0.2)
    elif deconv_activation == 'ELU':
        return ELU()
    return ReLU()


class Model:
    def __init__(
        self,
        input_tensor,
        cout = 1,
        num_layers = 6,
        num_initial_filters = 16,
        output_mask_logit = False,
        logging = False,
        dropout = 0.5,
        training = True,
    ):
        conv_activation_layer = _get_conv_activation_layer({})
        deconv_activation_layer = _get_deconv_activation_layer({})
        kernel_initializer = he_uniform(seed = 50)

        conv2d_factory = partial(
            Conv2D,
            strides = (2, 2),
            padding = 'same',
            kernel_initializer = kernel_initializer,
        )

        conv2d_transpose_factory = partial(
            Conv2DTranspose,
            strides = (2, 2),
            padding = 'same',
            kernel_initializer = kernel_initializer,
        )

        def resnet_block(input_tensor, filter_size):

            res = conv2d_factory(
                filter_size, (1, 1), strides = (1, 1), use_bias = False
            )(input_tensor)
            conv1 = conv2d_factory(filter_size, (5, 5), strides = (1, 1))(
                input_tensor
            )
            batch1 = BatchNormalization(axis = -1)(conv1, training = training)
            rel1 = conv_activation_layer(batch1)
            conv2 = conv2d_factory(filter_size, (5, 5), strides = (1, 1))(rel1)
            batch2 = BatchNormalization(axis = -1)(conv2, training = training)
            resconnection = Add()([res, batch2])
            rel2 = conv_activation_layer(resconnection)
            return MaxPooling2D(padding = 'same')(rel2)

        enc_outputs = []
        current_layer = input_tensor
        for i in range(num_layers):

            if i < num_layers - 1:
                current_layer = resnet_block(
                    current_layer, num_initial_filters * (2 ** i)
                )
                enc_outputs.append(current_layer)
            else:
                current_layer = conv2d_factory(
                    num_initial_filters * (2 ** i), (5, 5)
                )(current_layer)

            if logging:
                print(current_layer)

        for i in range(num_layers - 1):

            current_layer = conv2d_transpose_factory(
                num_initial_filters * (2 ** (num_layers - i - 2)), (5, 5)
            )((current_layer))
            current_layer = deconv_activation_layer(current_layer)
            current_layer = BatchNormalization(axis = -1)(
                current_layer, training = training
            )
            if i < 3:
                current_layer = Dropout(dropout)(
                    current_layer, training = training
                )
            current_layer = Concatenate(axis = -1)(
                [enc_outputs[-i - 1], current_layer]
            )
            if logging:
                print(current_layer)

        current_layer = conv2d_transpose_factory(1, (5, 5), strides = (2, 2))(
            (current_layer)
        )
        current_layer = deconv_activation_layer(current_layer)
        current_layer = BatchNormalization(axis = -1)(
            current_layer, training = training
        )

        if not output_mask_logit:
            last = Conv2D(
                cout,
                (4, 4),
                dilation_rate = (2, 2),
                activation = 'sigmoid',
                padding = 'same',
                kernel_initializer = kernel_initializer,
            )((current_layer))
            output = Multiply()([last, input_tensor])
            self.logits = output
        else:
            self.logits = Conv2D(
                cout,
                (4, 4),
                dilation_rate = (2, 2),
                padding = 'same',
                kernel_initializer = kernel_initializer,
            )((current_layer))
