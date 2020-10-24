import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Reshape
from tensorflow.keras.layers import (
    Conv2D,
    ZeroPadding2D,
    MaxPooling2D,
    AveragePooling2D,
)
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Lambda, Activation
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


def conv_bn_pool(
    inp_tensor,
    layer_idx,
    conv_filters,
    conv_kernel_size,
    conv_strides,
    conv_pad,
    pool = '',
    pool_size = (2, 2),
    pool_strides = None,
    conv_layer_prefix = 'conv',
):
    x = ZeroPadding2D(padding = conv_pad, name = 'pad{}'.format(layer_idx))(
        inp_tensor
    )
    x = Conv2D(
        filters = conv_filters,
        kernel_size = conv_kernel_size,
        strides = conv_strides,
        padding = 'valid',
        name = '{}{}'.format(conv_layer_prefix, layer_idx),
    )(x)
    x = BatchNormalization(
        epsilon = 1e-5, momentum = 1.0, name = 'bn{}'.format(layer_idx)
    )(x)
    x = Activation('relu', name = 'relu{}'.format(layer_idx))(x)
    if pool == 'max':
        x = MaxPooling2D(
            pool_size = pool_size,
            strides = pool_strides,
            name = 'mpool{}'.format(layer_idx),
        )(x)
    elif pool == 'avg':
        x = AveragePooling2D(
            pool_size = pool_size,
            strides = pool_strides,
            name = 'apool{}'.format(layer_idx),
        )(x)
    return x


# Block of layers: Conv --> BatchNorm --> ReLU --> Dynamic average pool (fc6 -> apool6 only)
def conv_bn_dynamic_apool(
    inp_tensor,
    layer_idx,
    conv_filters,
    conv_kernel_size,
    conv_strides,
    conv_pad,
    conv_layer_prefix = 'conv',
):
    x = ZeroPadding2D(padding = conv_pad, name = 'pad{}'.format(layer_idx))(
        inp_tensor
    )
    x = Conv2D(
        filters = conv_filters,
        kernel_size = conv_kernel_size,
        strides = conv_strides,
        padding = 'valid',
        name = '{}{}'.format(conv_layer_prefix, layer_idx),
    )(x)
    x = BatchNormalization(
        epsilon = 1e-5, momentum = 1.0, name = 'bn{}'.format(layer_idx)
    )(x)
    x = Activation('relu', name = 'relu{}'.format(layer_idx))(x)
    x = GlobalAveragePooling2D(name = 'gapool{}'.format(layer_idx))(x)
    x = Reshape((1, 1, conv_filters), name = 'reshape{}'.format(layer_idx))(x)
    return x


class MODEL(Model):
    def __init__(self, params = None, is_training = False):
        super(MODEL, self).__init__()

    def call(self, inputs, training = None, mask = None):
        inp = inputs['features_input']
        x = conv_bn_pool(
            inp,
            layer_idx = 1,
            conv_filters = 96,
            conv_kernel_size = (7, 7),
            conv_strides = (2, 2),
            conv_pad = (1, 1),
            pool = 'max',
            pool_size = (3, 3),
            pool_strides = (2, 2),
        )
        x = conv_bn_pool(
            x,
            layer_idx = 2,
            conv_filters = 256,
            conv_kernel_size = (5, 5),
            conv_strides = (2, 2),
            conv_pad = (1, 1),
            pool = 'max',
            pool_size = (3, 3),
            pool_strides = (2, 2),
        )
        x = conv_bn_pool(
            x,
            layer_idx = 3,
            conv_filters = 384,
            conv_kernel_size = (3, 3),
            conv_strides = (1, 1),
            conv_pad = (1, 1),
        )
        x = conv_bn_pool(
            x,
            layer_idx = 4,
            conv_filters = 256,
            conv_kernel_size = (3, 3),
            conv_strides = (1, 1),
            conv_pad = (1, 1),
        )
        x = conv_bn_pool(
            x,
            layer_idx = 5,
            conv_filters = 256,
            conv_kernel_size = (3, 3),
            conv_strides = (1, 1),
            conv_pad = (1, 1),
            pool = 'max',
            pool_size = (5, 3),
            pool_strides = (3, 2),
        )
        x = conv_bn_dynamic_apool(
            x,
            layer_idx = 6,
            conv_filters = 4096,
            conv_kernel_size = (9, 1),
            conv_strides = (1, 1),
            conv_pad = (0, 0),
            conv_layer_prefix = 'fc',
        )
        x = conv_bn_pool(
            x,
            layer_idx = 7,
            conv_filters = 1024,
            conv_kernel_size = (1, 1),
            conv_strides = (1, 1),
            conv_pad = (0, 0),
            conv_layer_prefix = 'fc',
        )
        x = Lambda(lambda y: K.l2_normalize(y, axis = 3), name = 'norm')(x)
        x = Conv2D(
            filters = 1024,
            kernel_size = (1, 1),
            strides = (1, 1),
            padding = 'valid',
            name = 'fc8',
        )(x)
        return x


class Model:
    def __init__(self, inputs, num_class, mode = 'train'):

        model = MODEL(is_training = mode == 'train')
        inputs = {'features_input': inputs}

        logits = model.call(inputs)
        logits = logits[:, 0, 0, :]
        self.logits = tf.layers.dense(logits, num_class)
