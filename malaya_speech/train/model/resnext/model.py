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
from . import layer as lay


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
        ksize = 3,
        num_layers = 6,
        num_initial_filters = 16,
        output_mask_logit = False,
        logging = False,
        dropout = 0.5,
        training = False,
    ):
        self.customlayers = lay.Layers()
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
            print(current_layer)
            if i < num_layers - 1:
                current_layer = self.residual_X(
                    current_layer,
                    ksize,
                    inchannel = int(current_layer.shape[-1]),
                    outchannel = num_initial_filters * (2 ** i),
                    name = f'residual_x_{i}',
                    logging = logging,
                    training = training,
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

    def residual_X(
        self,
        input,
        ksize,
        inchannel,
        outchannel,
        name = '',
        logging = False,
        training = True,
    ):
        convtmp_1 = self.customlayers.conv2d(
            input,
            self.customlayers.get_weight(
                vshape = [1, 1, inchannel, inchannel], name = '%s_1' % (name)
            ),
            stride_size = 1,
            padding = 'SAME',
        )
        convtmp_1bn = BatchNormalization(axis = -1)(
            convtmp_1, training = training
        )
        convtmp_1act = self.customlayers.elu(convtmp_1bn)
        convtmp_2 = self.customlayers.conv2d(
            convtmp_1act,
            self.customlayers.get_weight(
                vshape = [ksize, ksize, inchannel, inchannel],
                name = '%s_2' % (name),
            ),
            stride_size = 1,
            padding = 'SAME',
        )
        convtmp_2bn = BatchNormalization(axis = -1)(
            convtmp_2, training = training
        )
        convtmp_2act = self.customlayers.elu(convtmp_2bn)
        convtmp_3 = self.customlayers.conv2d(
            convtmp_2act,
            self.customlayers.get_weight(
                vshape = [ksize, ksize, inchannel, outchannel],
                name = '%s_3' % (name),
            ),
            stride_size = 1,
            padding = 'SAME',
        )
        convtmp_3bn = BatchNormalization(axis = -1)(
            convtmp_3, training = training
        )
        convtmp_3act = self.customlayers.elu(convtmp_3bn)

        if input.shape[-1] != convtmp_3act.shape[-1]:
            convtmp_sc = self.customlayers.conv2d(
                input,
                self.customlayers.get_weight(
                    vshape = [1, 1, inchannel, outchannel],
                    name = '%s_sc' % (name),
                ),
                stride_size = 1,
                padding = 'SAME',
            )
            convtmp_scbn = BatchNormalization(axis = -1)(
                convtmp_sc, training = training
            )
            convtmp_scact = self.customlayers.elu(convtmp_scbn)
            input = convtmp_scact

        output = input + convtmp_3act
        output = MaxPooling2D(padding = 'same')(output)

        if logging:
            print(name, output.shape)
        return output
