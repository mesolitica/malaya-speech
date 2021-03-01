import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv3D,
    Conv3DTranspose,
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
        cout = 1,
        output_name = 'output',
        params = {},
        output_mask_logit = False,
        dropout = 0.5,
        training = True,
        kernel_size = 4,
    ):
        conv_n_filters = params.get(
            'conv_n_filters', [16, 32, 64, 128, 256, 512]
        )
        conv_activation_layer = _get_conv_activation_layer(params)
        deconv_activation_layer = _get_deconv_activation_layer(params)
        kernel_initializer = he_uniform(seed = 50)
        conv3d_factory = partial(
            Conv3D,
            strides = (1, 2, 2),
            padding = 'same',
            kernel_initializer = kernel_initializer,
        )
        conv1 = conv3d_factory(
            conv_n_filters[0], (kernel_size, kernel_size, kernel_size)
        )(input_tensor)
        batch1 = BatchNormalization(axis = -1)(conv1, training = training)
        rel1 = conv_activation_layer(batch1)
        conv2 = conv3d_factory(
            conv_n_filters[1], (kernel_size, kernel_size, kernel_size)
        )(rel1)
        batch2 = BatchNormalization(axis = -1)(conv2, training = training)
        rel2 = conv_activation_layer(batch2)
        conv3 = conv3d_factory(
            conv_n_filters[2], (kernel_size, kernel_size, kernel_size)
        )(rel2)
        batch3 = BatchNormalization(axis = -1)(conv3, training = training)
        rel3 = conv_activation_layer(batch3)
        conv4 = conv3d_factory(
            conv_n_filters[3], (kernel_size, kernel_size, kernel_size)
        )(rel3)
        batch4 = BatchNormalization(axis = -1)(conv4, training = training)
        rel4 = conv_activation_layer(batch4)
        conv5 = conv3d_factory(
            conv_n_filters[4], (kernel_size, kernel_size, kernel_size)
        )(rel4)
        batch5 = BatchNormalization(axis = -1)(conv5, training = training)
        rel5 = conv_activation_layer(batch5)
        conv6 = conv3d_factory(
            conv_n_filters[5], (kernel_size, kernel_size, kernel_size)
        )(rel5)
        batch6 = BatchNormalization(axis = -1)(conv6, training = training)
        _ = conv_activation_layer(batch6)
        conv3d_transpose_factory = partial(
            Conv3DTranspose,
            strides = (1, 2, 2),
            padding = 'same',
            kernel_initializer = kernel_initializer,
        )
        up1 = conv3d_transpose_factory(
            conv_n_filters[4], (kernel_size, kernel_size, kernel_size)
        )((conv6))
        up1 = deconv_activation_layer(up1)
        batch7 = BatchNormalization(axis = -1)(up1, training = training)
        drop1 = Dropout(dropout)(batch7, training = training)
        merge1 = Concatenate(axis = -1)([conv5, drop1])
        up2 = conv3d_transpose_factory(
            conv_n_filters[3], (kernel_size, kernel_size, kernel_size)
        )((merge1))
        up2 = deconv_activation_layer(up2)
        batch8 = BatchNormalization(axis = -1)(up2, training = training)
        drop2 = Dropout(dropout)(batch8, training = training)
        merge2 = Concatenate(axis = -1)([conv4, drop2])
        up3 = conv3d_transpose_factory(
            conv_n_filters[2], (kernel_size, kernel_size, kernel_size)
        )((merge2))
        up3 = deconv_activation_layer(up3)
        batch9 = BatchNormalization(axis = -1)(up3, training = training)
        drop3 = Dropout(dropout)(batch9, training = training)
        merge3 = Concatenate(axis = -1)([conv3, drop3])
        up4 = conv3d_transpose_factory(
            conv_n_filters[1], (kernel_size, kernel_size, kernel_size)
        )((merge3))
        up4 = deconv_activation_layer(up4)
        batch10 = BatchNormalization(axis = -1)(up4, training = training)
        merge4 = Concatenate(axis = -1)([conv2, batch10])
        up5 = conv3d_transpose_factory(
            conv_n_filters[0], (kernel_size, kernel_size, kernel_size)
        )((merge4))
        up5 = deconv_activation_layer(up5)
        batch11 = BatchNormalization(axis = -1)(up5, training = training)
        merge5 = Concatenate(axis = -1)([conv1, batch11])
        up6 = conv3d_transpose_factory(
            1, (kernel_size, kernel_size, kernel_size), strides = (1, 2, 2)
        )((merge5))
        up6 = deconv_activation_layer(up6)
        batch12 = BatchNormalization(axis = -1)(up6, training = training)
        if not output_mask_logit:
            up7 = Conv3D(
                cout,
                (kernel_size, kernel_size, kernel_size),
                dilation_rate = (1, 2, 2),
                activation = 'sigmoid',
                padding = 'same',
                kernel_initializer = kernel_initializer,
            )((batch12))
            output = Multiply(name = output_name)([up7, input_tensor])
            self.logits = output
        else:
            self.logits = Conv3D(
                cout,
                (kernel_size, kernel_size, kernel_size),
                dilation_rate = (1, 2, 2),
                padding = 'same',
                kernel_initializer = kernel_initializer,
            )((batch12))
