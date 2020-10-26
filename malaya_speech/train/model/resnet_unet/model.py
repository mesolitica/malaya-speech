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
            strides = (1, 1),
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

            res = conv2d_factory(filter_size, (1, 1), use_bias = False)(
                input_tensor
            )
            conv1 = conv2d_factory(filter_size, (5, 5))(input_tensor)
            batch1 = BatchNormalization(axis = -1)(conv1)
            rel1 = conv_activation_layer(batch1)
            conv2 = conv2d_factory(filter_size, (5, 5))(rel1)
            batch2 = BatchNormalization(axis = -1)(conv2)
            resconnection = Add()([res, batch2])
            rel2 = conv_activation_layer(resconnection)
            return rel2, MaxPooling2D(padding = 'same')(rel2)

        def de_resnet_block(left_tensor, right_tensor, filter_size):

            up = conv2d_transpose_factory(filter_size, (5, 5))((left_tensor))
            merged = Concatenate(axis = -1)([up, right_tensor])

            res = conv2d_factory(filter_size, (1, 1), use_bias = False)(merged)
            conv1 = conv2d_factory(filter_size, (5, 5))(merged)
            batch1 = BatchNormalization(axis = -1)(conv1)
            rel1 = deconv_activation_layer(batch1)
            conv2 = conv2d_factory(filter_size, (5, 5))(rel1)
            batch2 = BatchNormalization(axis = -1)(conv2)
            resconnection = Add()([res, batch2])
            rel2 = deconv_activation_layer(resconnection)
            return rel2

        # print(input_tensor.shape)
        conv1, rel1 = resnet_block(input_tensor, conv_n_filters[0])
        # print(conv1.shape, rel1.shape)
        conv2, rel2 = resnet_block(rel1, conv_n_filters[1])
        # print(conv2.shape, rel2.shape)
        conv3, rel3 = resnet_block(rel2, conv_n_filters[2])
        # print(conv3.shape, rel3.shape)
        conv4, rel4 = resnet_block(rel3, conv_n_filters[3])
        # print(conv4.shape, rel4.shape)
        conv5, rel5 = resnet_block(rel4, conv_n_filters[4])
        # print(conv5.shape, rel5.shape)
        conv6, _ = resnet_block(rel5, conv_n_filters[5])
        # print(conv6.shape)

        merge1 = Dropout(0.5)(de_resnet_block(conv6, conv5, conv_n_filters[4]))
        # print(merge1.shape)
        merge2 = Dropout(0.5)(de_resnet_block(merge1, conv4, conv_n_filters[3]))
        # print(merge2.shape)
        merge3 = Dropout(0.5)(de_resnet_block(merge2, conv3, conv_n_filters[2]))
        # print(merge3.shape)
        merge4 = de_resnet_block(merge3, conv2, conv_n_filters[1])
        # print(merge4.shape)
        merge5 = de_resnet_block(merge4, conv1, conv_n_filters[0])
        # print(merge5.shape)

        if not output_mask_logit:
            up7 = Conv2D(
                1,
                (4, 4),
                dilation_rate = (2, 2),
                activation = 'sigmoid',
                padding = 'same',
                kernel_initializer = kernel_initializer,
            )((merge5))
            output = Multiply(name = output_name)([up7, input_tensor])
            self.logits = output
        else:
            self.logits = Conv2D(
                1,
                (4, 4),
                dilation_rate = (2, 2),
                padding = 'same',
                kernel_initializer = kernel_initializer,
            )((merge5))
