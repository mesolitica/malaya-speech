import tensorflow as tf
from ..openseq2seq.model import TDNNEncoder


residual_dense = False

encoder_config = {
    'convnet_layers': [
        {
            'type': 'sep_conv1d',
            'repeat': 1,
            'kernel_size': [11],
            'stride': [2],
            'num_channels': 256,
            'padding': 'SAME',
            'dilation': [1],
        },
        {
            'type': 'sep_conv1d',
            'repeat': 3,
            'kernel_size': [11],
            'stride': [1],
            'num_channels': 256,
            'padding': 'SAME',
            'dilation': [1],
            'residual': True,
            'residual_dense': residual_dense,
        },
        {
            'type': 'sep_conv1d',
            'repeat': 3,
            'kernel_size': [11],
            'stride': [1],
            'num_channels': 256,
            'padding': 'SAME',
            'dilation': [1],
            'residual': True,
            'residual_dense': residual_dense,
        },
        {
            'type': 'sep_conv1d',
            'repeat': 3,
            'kernel_size': [13],
            'stride': [1],
            'num_channels': 256,
            'padding': 'SAME',
            'dilation': [1],
            'residual': True,
            'residual_dense': residual_dense,
        },
        {
            'type': 'sep_conv1d',
            'repeat': 3,
            'kernel_size': [13],
            'stride': [1],
            'num_channels': 256,
            'padding': 'SAME',
            'dilation': [1],
            'residual': True,
            'residual_dense': residual_dense,
        },
        {
            'type': 'sep_conv1d',
            'repeat': 3,
            'kernel_size': [17],
            'stride': [1],
            'num_channels': 512,
            'padding': 'SAME',
            'dilation': [1],
            'residual': True,
            'residual_dense': residual_dense,
        },
        {
            'type': 'sep_conv1d',
            'repeat': 3,
            'kernel_size': [17],
            'stride': [1],
            'num_channels': 512,
            'padding': 'SAME',
            'dilation': [1],
            'residual': True,
            'residual_dense': residual_dense,
        },
        {
            'type': 'sep_conv1d',
            'repeat': 3,
            'kernel_size': [21],
            'stride': [1],
            'num_channels': 512,
            'padding': 'SAME',
            'dilation': [1],
            'residual': True,
            'residual_dense': residual_dense,
        },
        {
            'type': 'sep_conv1d',
            'repeat': 3,
            'kernel_size': [21],
            'stride': [1],
            'num_channels': 512,
            'padding': 'SAME',
            'dilation': [1],
            'residual': True,
            'residual_dense': residual_dense,
        },
        {
            'type': 'sep_conv1d',
            'repeat': 3,
            'kernel_size': [25],
            'stride': [1],
            'num_channels': 512,
            'padding': 'SAME',
            'dilation': [1],
            'residual': True,
            'residual_dense': residual_dense,
        },
        {
            'type': 'sep_conv1d',
            'repeat': 3,
            'kernel_size': [25],
            'stride': [1],
            'num_channels': 512,
            'padding': 'SAME',
            'dilation': [1],
            'residual': True,
            'residual_dense': residual_dense,
        },
        {
            'type': 'sep_conv1d',
            'repeat': 1,
            'kernel_size': [29],
            'stride': [1],
            'num_channels': 512,
            'padding': 'SAME',
            'dilation': [2],
        },
        {
            'type': 'sep_conv1d',
            'repeat': 1,
            'kernel_size': [1],
            'stride': [1],
            'num_channels': 1024,
            'padding': 'SAME',
            'dilation': [1],
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
    def __init__(self, inputs, inputs_length, training = True):
        if training:
            mode = 'train'
        else:
            mode = 'eval'
        self.model = TDNNEncoder(config, None, mode = mode)
        input_dict = {'source_tensors': [inputs, inputs_length]}
        self.logits = self.model.encode(input_dict)
