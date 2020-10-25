import tensorflow as tf
from . import layer, abstract

residual_dense = False

config = {
    'convnet_layers': [
        {
            'type': 'sep_conv1d',
            'repeat': 1,
            'kernel_size': [3],
            'stride': [1],
            'num_channels': 512,
            'padding': 'SAME',
            'dilation': [1],
            'dropout_keep_prob': 0.5,
            'residual': True,
        },
        # 2
        {
            'type': 'sep_conv1d',
            'repeat': 2,
            'kernel_size': [7],
            'stride': [1],
            'num_channels': 512,
            'padding': 'SAME',
            'dilation': [1],
            'dropout_keep_prob': 0.5,
            'residual': True,
        },
        {
            'type': 'sep_conv1d',
            'repeat': 2,
            'kernel_size': [11],
            'stride': [1],
            'num_channels': 512,
            'padding': 'SAME',
            'dilation': [1],
            'dropout_keep_prob': 0.5,
            'residual': True,
        },
        # 4
        {
            'type': 'sep_conv1d',
            'repeat': 2,
            'kernel_size': [15],
            'stride': [1],
            'num_channels': 512,
            'padding': 'SAME',
            'dilation': [1],
            'dropout_keep_prob': 0.5,
            'residual': True,
        },
        {
            'type': 'sep_conv1d',
            'repeat': 1,
            'kernel_size': [1],
            'stride': [1],
            'num_channels': 1500,
            'padding': 'SAME',
            'dilation': [1],
            'dropout_keep_prob': 1.0,
            'residual': False,
        },
    ],
    'dropout_keep_prob': 1.0,
    'initializer': tf.contrib.layers.xavier_initializer,
    'initializer_params': {'uniform': False},
    'normalization': 'batch_norm',
    'activation_fn': tf.nn.relu,
    'data_format': 'channels_last',
    'use_conv_mask': True,
}


class Model:
    def __init__(self, inputs, inputs_length, num_class = 7205, mode = 'train'):
        self.model = abstract.TDNNEncoder(config, None, mode = mode)
        input_dict = {'source_tensors': [inputs, inputs_length]}
        logits = self.model.encode(input_dict)['outputs']

        def affine(x, output_size):
            f = tf.layers.dense(x, 512)
            f = tf.compat.v1.layers.batch_normalization(f)
            return tf.nn.relu(f)

        pooled = tf.concat(
            [
                tf.math.reduce_mean(logits, axis = 1),
                tf.math.reduce_std(logits, axis = 1),
            ],
            axis = 1,
        )
        f = affine(pooled, 512)
        f = affine(f, 512)
        self.logits = tf.layers.dense(f, num_class, use_bias = False)
