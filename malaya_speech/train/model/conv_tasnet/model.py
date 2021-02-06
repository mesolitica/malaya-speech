# https://github.com/gantheory/Conv-TasNet/blob/master/main.py

import tensorflow as tf
import numpy as np


class Model:
    def __init__(
        self,
        input_tensor,
        C = 1,
        N = 256,
        L = 20,
        B = 256,
        H = 512,
        P = 3,
        X = 8,
        R = 4,
    ):
        self.H = H
        self.P = P
        self.dtype = tf.float32
        layers = {
            'conv1d_encoder': tf.keras.layers.Conv1D(
                filters = N,
                kernel_size = L,
                strides = L // 2,
                activation = tf.nn.relu,
                name = 'encode_conv1d',
                padding = 'same',
            ),
            'bottleneck': tf.keras.layers.Conv1D(B, 1, 1),
            '1d_deconv': tf.keras.layers.Dense(L // 2, use_bias = False),
        }
        for i in range(C):
            layers['1x1_conv_decoder_{}'.format(i)] = tf.keras.layers.Conv1D(
                N, 1, 1
            )
        for r in range(R):
            for x in range(X):
                now_block = 'block_{}_{}_'.format(r, x)
                layers[now_block + 'first_1x1_conv'] = tf.keras.layers.Conv1D(
                    filters = H, kernel_size = 1
                )
                layers[now_block + 'first_PReLU'] = tf.keras.layers.PReLU(
                    shared_axes = [1]
                )
                layers[now_block + 'second_PReLU'] = tf.keras.layers.PReLU(
                    shared_axes = [1]
                )
                layers[now_block + 'second_1x1_conv'] = tf.keras.layers.Conv1D(
                    filters = B, kernel_size = 1
                )

        with tf.variable_scope('encoder'):
            encoded_input = layers['conv1d_encoder'](inputs = input_tensor)
        with tf.variable_scope('bottleneck'):
            norm_input = self._channel_norm(encoded_input, 'bottleneck')
            block_input = layers['bottleneck'](norm_input)
        for r in range(R):
            for x in range(X):
                now_block = 'block_{}_{}_'.format(r, x)
                with tf.variable_scope(now_block):
                    block_output = layers[now_block + 'first_1x1_conv'](
                        block_input
                    )
                    block_output = layers[now_block + 'first_PReLU'](
                        block_output
                    )
                    block_output = self._global_norm(block_output, 'first')
                    block_output = self._depthwise_conv1d(block_output, x)
                    block_output = layers[now_block + 'second_PReLU'](
                        block_output
                    )
                    block_output = self._global_norm(block_output, 'second')
                    block_output = layers[now_block + 'second_1x1_conv'](
                        block_output
                    )
                    block_input = block_output = block_output + block_input

        sep_output_list = [
            layers['1x1_conv_decoder_{}'.format(i)](block_output)
            for i in range(C)
        ]
        sep_output_list = [
            layers['1d_deconv'](sep_output) for sep_output in sep_output_list
        ]
        self.logits = outputs = [
            tf.signal.overlap_and_add(signal = sep_output, frame_step = L // 2)
            for sep_output in sep_output_list
        ]

    def _channel_norm(self, inputs, name):
        # inputs: [batch_size, some len, channel_size]
        with tf.variable_scope(name):
            channel_size = inputs.shape[-1]

            E = tf.reshape(
                tf.reduce_mean(inputs, axis = [2]), [-1, tf.shape(inputs)[1], 1]
            )
            Var = tf.reshape(
                tf.reduce_mean((inputs - E) ** 2, axis = [2]),
                [-1, tf.shape(inputs)[1], 1],
            )
            gamma = tf.get_variable(
                'gamma', shape = [1, 1, channel_size], dtype = self.dtype
            )
            beta = tf.get_variable(
                'beta', shape = [1, 1, channel_size], dtype = self.dtype
            )
            return ((inputs - E) / (Var + 1e-8) ** 0.5) * gamma + beta

    def _global_norm(self, inputs, name):
        with tf.variable_scope(name):
            channel_size = inputs.shape[-1]
            E = tf.reshape(tf.reduce_mean(inputs, axis = [1, 2]), [-1, 1, 1])
            Var = tf.reshape(
                tf.reduce_mean((inputs - E) ** 2, axis = [1, 2]), [-1, 1, 1]
            )
            gamma = tf.get_variable(
                'gamma', shape = [1, 1, channel_size], dtype = self.dtype
            )
            beta = tf.get_variable(
                'beta', shape = [1, 1, channel_size], dtype = self.dtype
            )
            return ((inputs - E) / (Var + 1e-8) ** 0.5) * gamma + beta

    def _depthwise_conv1d(self, inputs, x):
        l = tf.shape(inputs)[1]
        inputs = tf.reshape(inputs, [-1, 1, l, self.H])
        filters = tf.get_variable(
            'dconv_filters', [1, self.P, self.H, 1], dtype = self.dtype
        )
        outputs = tf.nn.depthwise_conv2d(
            input = inputs,
            filter = filters,
            strides = [1, 1, 1, 1],
            padding = 'SAME',
            rate = [1, 2 ** x],
        )
        return tf.reshape(outputs, [-1, l, self.H])
