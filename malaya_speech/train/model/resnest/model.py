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
        radix = 4,
        kpaths = 4,
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
                current_layer = self.residual_S(
                    current_layer,
                    ksize,
                    inchannel = int(current_layer.shape[-1]),
                    outchannel = num_initial_filters * (2 ** i),
                    radix = radix,
                    kpaths = kpaths,
                    name = f'residual_s_{i}',
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

    def residual_S(
        self,
        input,
        ksize,
        inchannel,
        outchannel,
        radix,
        kpaths,
        name = '',
        logging = False,
        training = True,
    ):
        convtmp_1 = self.customlayers.conv2d(
            input,
            self.customlayers.get_weight(
                vshape = [ksize, ksize, inchannel, outchannel],
                name = '%s_1' % (name),
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
                vshape = [ksize, ksize, outchannel, outchannel],
                name = '%s_2' % (name),
            ),
            stride_size = 1,
            padding = 'SAME',
        )
        convtmp_2bn = BatchNormalization(axis = -1)(
            convtmp_2, training = training
        )
        convtmp_2act = self.customlayers.elu(convtmp_2bn)

        concats_1 = None
        for idx_k in range(kpaths):
            cardinal = self.cardinal(
                convtmp_2act,
                ksize,
                outchannel,
                outchannel,
                radix,
                kpaths,
                name = '%s_car_k%d' % (name, idx_k),
                training = training,
            )
            if idx_k == 0:
                concats_1 = cardinal
            else:
                concats_1 = tf.concat([concats_1, cardinal], axis = 3)
        concats_2 = self.customlayers.conv2d(
            concats_1,
            self.customlayers.get_weight(
                vshape = [1, 1, outchannel, outchannel], name = '%s_cc' % (name)
            ),
            stride_size = 1,
            padding = 'SAME',
        )
        concats_2 = concats_2 + convtmp_2act

        if input.shape[-1] != concats_2.shape[-1]:
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

        output = input + concats_2
        output = MaxPooling2D(padding = 'same')(output)
        if logging:
            print(name, output.shape)
        return output

    def cardinal(
        self,
        input,
        ksize,
        inchannel,
        outchannel,
        radix,
        kpaths,
        name = '',
        logging = False,
        training = True,
    ):

        if logging:
            print('cardinal')
        outchannel_cv11 = int(outchannel / radix / kpaths)
        outchannel_cvkk = int(outchannel / kpaths)

        inputs = []
        for idx_r in range(radix):
            conv1 = self.customlayers.conv2d(
                input,
                self.customlayers.get_weight(
                    vshape = [1, 1, inchannel, outchannel_cv11],
                    name = '%s1_r%d' % (name, idx_r),
                ),
                stride_size = 1,
                padding = 'SAME',
            )
            conv1_bn = BatchNormalization(axis = -1)(conv1, training = training)
            conv1_act = self.customlayers.elu(conv1_bn)

            conv2 = self.customlayers.conv2d(
                conv1_act,
                self.customlayers.get_weight(
                    vshape = [ksize, ksize, outchannel_cv11, outchannel_cvkk],
                    name = '%s2_r%d' % (name, idx_r),
                ),
                stride_size = 1,
                padding = 'SAME',
            )
            conv2_bn = BatchNormalization(axis = -1)(conv2, training = training)
            conv2_act = self.customlayers.elu(conv2_bn)
            inputs.append(conv2_act)

        return self.split_attention(
            inputs,
            outchannel_cvkk,
            name = '%s_att' % (name),
            training = training,
        )

    def split_attention(
        self, inputs, inchannel, name = '', verbose = False, training = True
    ):

        if verbose:
            print('split attention')
        radix = len(inputs)
        input_holder = None
        for idx_i, input in enumerate(inputs):
            if idx_i == 0:
                input_holder = input
            else:
                input_holder += input

        ga_pool = tf.math.reduce_mean(input_holder, axis = (1, 2))
        ga_pool = tf.expand_dims(tf.expand_dims(ga_pool, axis = 1), axis = 1)

        dense1 = self.customlayers.conv2d(
            ga_pool,
            self.customlayers.get_weight(
                vshape = [1, 1, inchannel, inchannel], name = '%s1' % (name)
            ),
            stride_size = 1,
            padding = 'SAME',
        )
        dense1_bn = BatchNormalization(axis = -1)(dense1, training = training)
        dense1_act = self.customlayers.elu(dense1_bn)

        output_holder = None
        for idx_r in range(radix):
            dense2 = self.customlayers.conv2d(
                dense1_act,
                self.customlayers.get_weight(
                    vshape = [1, 1, inchannel, inchannel],
                    name = '%s2_r%d' % (name, idx_r),
                ),
                stride_size = 1,
                padding = 'SAME',
            )
            if radix == 1:
                r_softmax = self.customlayers.sigmoid(dense2)
            elif radix > 1:
                r_softmax = self.customlayers.softmax(dense2)

            if idx_r == 0:
                output_holder = inputs[idx_r] * r_softmax
            else:
                output_holder += inputs[idx_r] * r_softmax

        return output_holder
