import tensorflow as tf


class Model:
    def __init__(self, inputs, vocab_size = 256):
        self.X = inputs
        self.X = tf.expand_dims(self.X, -1)
        batch_size = tf.shape(self.X)[0]
        filters = [128, 128, 256, 256, 256]
        strides = [1, 2]
        x = self.conv2d(self.X, 'cnn-1', 3, filters[0], strides[0])
        x = self.batch_norm('bn1', x)
        x = self.leaky_relu(x, 0.01)
        x = self.max_pool(x, 2, strides[1])
        x = self.conv2d(x, 'cnn-2', 3, filters[1], strides[0])
        x = self.batch_norm('bn2', x)
        x = self.leaky_relu(x, 0.01)
        x = self.max_pool(x, 2, strides[1])
        x = self.conv2d(x, 'cnn-3', 3, filters[2], strides[0])
        x = self.batch_norm('bn3', x)
        x = self.leaky_relu(x, 0.01)
        x = self.max_pool(x, 2, strides[1])
        x = self.conv2d(x, 'cnn-4', 3, filters[3], strides[0])
        x = self.batch_norm('bn4', x)
        x = self.leaky_relu(x, 0.01)
        x = self.max_pool(x, 2, strides[1])
        x = self.conv2d(x, 'cnn-5', 3, filters[4], strides[0])
        x = self.batch_norm('bn5', x)
        x = self.leaky_relu(x, 0.01)
        x = self.max_pool(x, 2, 1, padding = 'VALID')

        x = tf.reshape(x, [batch_size, -1, filters[4]])

        cell = tf.contrib.rnn.LSTMCell(num_hidden)
        cell1 = tf.contrib.rnn.LSTMCell(num_hidden)
        stack = tf.contrib.rnn.MultiRNNCell([cell, cell1])

        outputs, _ = tf.nn.dynamic_rnn(stack, x, dtype = tf.float32)
        self.logits = tf.layers.dense(outputs, vocab_size)

    def conv2d(self, x, name, filter_size, channel_out, strides):
        with tf.variable_scope(name):
            return tf.layers.conv2d(
                x, channel_out, filter_size, strides, padding = 'SAME'
            )

    def batch_norm(self, name, x):
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]
            beta = tf.get_variable(
                'beta',
                params_shape,
                tf.float32,
                initializer = tf.constant_initializer(0.0, tf.float32),
            )
            gamma = tf.get_variable(
                'gamma',
                params_shape,
                tf.float32,
                initializer = tf.constant_initializer(1.0, tf.float32),
            )
            mean, variance = tf.nn.moments(x, [0, 1, 2], name = 'moments')
            x_bn = tf.nn.batch_normalization(
                x, mean, variance, beta, gamma, 0.001
            )
            x_bn.set_shape(x.get_shape())
            return x_bn

    def leaky_relu(self, x, leak = 0):
        return tf.where(tf.less(x, 0.0), leak * x, x, name = 'leaky_relu')

    def max_pool(self, x, size, strides, padding = 'SAME'):
        return tf.nn.max_pool(
            x,
            ksize = [1, size, size, 1],
            strides = [1, strides, strides, 1],
            padding = padding,
            name = 'max_pool',
        )
