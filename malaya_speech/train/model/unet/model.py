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
        num_layers = 6,
        num_initial_filters = 16,
        output_mask_logit = False,
        logging = False,
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

        enc_outputs = []
        current_layer = input_tensor
        for i in range(num_layers):

            current_layer = conv2d_factory(
                num_initial_filters * (2 ** i), (5, 5)
            )(current_layer)
            if i < num_layers - 1:
                current_layer = BatchNormalization(axis = -1)(current_layer)
                current_layer = conv_activation_layer(current_layer)
                enc_outputs.append(current_layer)

            if logging:
                print(current_layer)

        for i in range(num_layers - 1):

            current_layer = conv2d_transpose_factory(
                num_initial_filters * (2 ** (num_layers - i - 2)), (5, 5)
            )((current_layer))
            current_layer = deconv_activation_layer(current_layer)
            current_layer = BatchNormalization(axis = -1)(current_layer)
            if i < 3:
                current_layer = Dropout(0.5)(current_layer)
            current_layer = Concatenate(axis = -1)(
                [enc_outputs[-i - 1], current_layer]
            )
            if logging:
                print(current_layer)

        current_layer = conv2d_transpose_factory(1, (5, 5), strides = (2, 2))(
            (current_layer)
        )
        current_layer = deconv_activation_layer(current_layer)
        current_layer = BatchNormalization(axis = -1)(current_layer)

        if not output_mask_logit:
            last = Conv2D(
                1,
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
                1,
                (4, 4),
                dilation_rate = (2, 2),
                padding = 'same',
                kernel_initializer = kernel_initializer,
            )((current_layer))
