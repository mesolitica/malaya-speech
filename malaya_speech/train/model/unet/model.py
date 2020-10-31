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
        output_name = 'output',
        params = {},
        output_mask_logit = False,
    ):
        conv_n_filters = params.get(
            'conv_n_filters', [16, 32, 64, 128, 256, 512]
        )
        conv_activation_layer = _get_conv_activation_layer(params)
        deconv_activation_layer = _get_deconv_activation_layer(params)
        kernel_initializer = he_uniform(seed = 50)
        conv2d_factory = partial(
            Conv2D,
            strides = (2, 2),
            padding = 'same',
            kernel_initializer = kernel_initializer,
        )
        conv1 = conv2d_factory(conv_n_filters[0], (5, 5))(input_tensor)
        batch1 = BatchNormalization(axis = -1)(conv1)
        rel1 = conv_activation_layer(batch1)
        conv2 = conv2d_factory(conv_n_filters[1], (5, 5))(rel1)
        batch2 = BatchNormalization(axis = -1)(conv2)
        rel2 = conv_activation_layer(batch2)
        conv3 = conv2d_factory(conv_n_filters[2], (5, 5))(rel2)
        batch3 = BatchNormalization(axis = -1)(conv3)
        rel3 = conv_activation_layer(batch3)
        conv4 = conv2d_factory(conv_n_filters[3], (5, 5))(rel3)
        batch4 = BatchNormalization(axis = -1)(conv4)
        rel4 = conv_activation_layer(batch4)
        conv5 = conv2d_factory(conv_n_filters[4], (5, 5))(rel4)
        batch5 = BatchNormalization(axis = -1)(conv5)
        rel5 = conv_activation_layer(batch5)
        conv6 = conv2d_factory(conv_n_filters[5], (5, 5))(rel5)
        batch6 = BatchNormalization(axis = -1)(conv6)
        _ = conv_activation_layer(batch6)
        conv2d_transpose_factory = partial(
            Conv2DTranspose,
            strides = (2, 2),
            padding = 'same',
            kernel_initializer = kernel_initializer,
        )
        up1 = conv2d_transpose_factory(conv_n_filters[4], (5, 5))((conv6))
        up1 = deconv_activation_layer(up1)
        batch7 = BatchNormalization(axis = -1)(up1)
        drop1 = Dropout(0.5)(batch7)
        merge1 = Concatenate(axis = -1)([conv5, drop1])
        up2 = conv2d_transpose_factory(conv_n_filters[3], (5, 5))((merge1))
        up2 = deconv_activation_layer(up2)
        batch8 = BatchNormalization(axis = -1)(up2)
        drop2 = Dropout(0.5)(batch8)
        merge2 = Concatenate(axis = -1)([conv4, drop2])
        up3 = conv2d_transpose_factory(conv_n_filters[2], (5, 5))((merge2))
        up3 = deconv_activation_layer(up3)
        batch9 = BatchNormalization(axis = -1)(up3)
        drop3 = Dropout(0.5)(batch9)
        merge3 = Concatenate(axis = -1)([conv3, drop3])
        up4 = conv2d_transpose_factory(conv_n_filters[1], (5, 5))((merge3))
        up4 = deconv_activation_layer(up4)
        batch10 = BatchNormalization(axis = -1)(up4)
        merge4 = Concatenate(axis = -1)([conv2, batch10])
        up5 = conv2d_transpose_factory(conv_n_filters[0], (5, 5))((merge4))
        up5 = deconv_activation_layer(up5)
        batch11 = BatchNormalization(axis = -1)(up5)
        merge5 = Concatenate(axis = -1)([conv1, batch11])
        up6 = conv2d_transpose_factory(1, (5, 5), strides = (2, 2))((merge5))
        up6 = deconv_activation_layer(up6)
        batch12 = BatchNormalization(axis = -1)(up6)
        if not output_mask_logit:
            up7 = Conv2D(
                1,
                (4, 4),
                dilation_rate = (2, 2),
                activation = 'sigmoid',
                padding = 'same',
                kernel_initializer = kernel_initializer,
            )((batch12))
            output = Multiply(name = output_name)([up7, input_tensor])
            self.logits = output
        else:
            self.logits = Conv2D(
                1,
                (4, 4),
                dilation_rate = (2, 2),
                padding = 'same',
                kernel_initializer = kernel_initializer,
            )((batch12))
