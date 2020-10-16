import tensorflow as tf
from .layer import *


class Discriminator:
    def __init__(self, inputs, targets, ndf = 64):
        n_layers = 3
        layers = []
        input = tf.concat([inputs, targets], axis = 3)
        with tf.variable_scope('layer_1'):
            convolved = discrim_conv(input, ndf, stride = 2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        for i in range(n_layers):
            with tf.variable_scope('layer_%d' % (len(layers) + 1)):
                out_channels = ndf * min(2 ** (i + 1), 8)
                stride = 1 if i == n_layers - 1 else 2
                convolved = discrim_conv(
                    layers[-1], out_channels, stride = stride
                )
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)
        with tf.variable_scope('layer_%d' % (len(layers) + 1)):
            convolved = discrim_conv(rectified, out_channels = 1, stride = 1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        self.logits = layers[-1]
