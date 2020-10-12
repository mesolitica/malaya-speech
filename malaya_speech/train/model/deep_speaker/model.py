import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda, Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

NUM_FBANKS = 64


class DeepSpeakerModel:
    def __init__(self):
        self.clipped_relu_count = 0

    def keras_model(self):
        return self.m

    def get_weights(self):
        w = self.m.get_weights()
        if self.include_softmax:
            w.pop()  # last 2 are the W_softmax and b_softmax.
            w.pop()
        return w

    def clipped_relu(self, inputs):
        relu = Lambda(
            lambda y: K.minimum(K.maximum(y, 0), 20),
            name = f'clipped_relu_{self.clipped_relu_count}',
        )(inputs)
        self.clipped_relu_count += 1
        return relu

    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
        conv_name_base = f'res{stage}_{block}_branch'

        x = Conv2D(
            filters,
            kernel_size = kernel_size,
            strides = 1,
            activation = None,
            padding = 'same',
            kernel_initializer = 'glorot_uniform',
            kernel_regularizer = regularizers.l2(l = 0.0001),
            name = conv_name_base + '_2a',
        )(input_tensor)
        x = BatchNormalization(name = conv_name_base + '_2a_bn')(x)
        x = self.clipped_relu(x)

        x = Conv2D(
            filters,
            kernel_size = kernel_size,
            strides = 1,
            activation = None,
            padding = 'same',
            kernel_initializer = 'glorot_uniform',
            kernel_regularizer = regularizers.l2(l = 0.0001),
            name = conv_name_base + '_2b',
        )(x)
        x = BatchNormalization(name = conv_name_base + '_2b_bn')(x)

        x = self.clipped_relu(x)

        x = layers.add([x, input_tensor])
        x = self.clipped_relu(x)
        return x

    def conv_and_res_block(self, inp, filters, stage):
        conv_name = 'conv{}-s'.format(filters)
        o = Conv2D(
            filters,
            kernel_size = 5,
            strides = 2,
            activation = None,
            padding = 'same',
            kernel_initializer = 'glorot_uniform',
            kernel_regularizer = regularizers.l2(l = 0.0001),
            name = conv_name,
        )(inp)
        o = BatchNormalization(name = conv_name + '_bn')(o)
        o = self.clipped_relu(o)
        for i in range(3):
            o = self.identity_block(
                o, kernel_size = 3, filters = filters, stage = stage, block = i
            )
        return o

    def cnn_component(self, inp):
        x = self.conv_and_res_block(inp, 64, stage = 1)
        x = self.conv_and_res_block(x, 128, stage = 2)
        x = self.conv_and_res_block(x, 256, stage = 3)
        x = self.conv_and_res_block(x, 512, stage = 4)
        return x

    def set_weights(self, w):
        for layer, layer_w in zip(self.m.layers, w):
            layer.set_weights(layer_w)
            logger.info(f'Setting weights for [{layer.name}]...')


class Model:
    def __init__(self, inputs, num_class, mode = 'train'):
        deepspeaker = DeepSpeakerModel()
        x = deepspeaker.cnn_component(inputs)
        x = Reshape((-1, 2048))(x)
        x = Lambda(lambda y: K.mean(y, axis = 1), name = 'average')(x)
        x = Dense(512, name = 'affine')(x)
        x = tf.layers.dense(x, num_class)
        self.logits = x
