# Copyright (c) 2018 NVIDIA Corporation

import tensorflow as tf
import copy
from .layer import (
    conv_actv,
    conv_bn_actv,
    conv_ln_actv,
    conv_in_actv,
    conv_bn_res_bn_actv,
)


def check_params(config, required_dict, optional_dict):
    if required_dict is None or optional_dict is None:
        return

    for pm, vals in required_dict.items():
        if pm not in config:
            raise ValueError('{} parameter has to be specified'.format(pm))
        else:
            if vals == str:
                vals = string_types
            if vals and isinstance(vals, list) and config[pm] not in vals:
                raise ValueError('{} has to be one of {}'.format(pm, vals))
            if (
                vals
                and not isinstance(vals, list)
                and not isinstance(config[pm], vals)
            ):
                raise ValueError('{} has to be of type {}'.format(pm, vals))

    for pm, vals in optional_dict.items():
        if vals == str:
            vals = string_types
        if pm in config:
            if vals and isinstance(vals, list) and config[pm] not in vals:
                raise ValueError('{} has to be one of {}'.format(pm, vals))
            if (
                vals
                and not isinstance(vals, list)
                and not isinstance(config[pm], vals)
            ):
                raise ValueError('{} has to be of type {}'.format(pm, vals))

    for pm in config:
        if pm not in required_dict and pm not in optional_dict:
            raise ValueError('Unknown parameter: {}'.format(pm))


def cast_types(input_dict, dtype):
    cast_input_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, tf.Tensor):
            if value.dtype == tf.float16 or value.dtype == tf.float32:
                if value.dtype.base_dtype != dtype.base_dtype:
                    cast_input_dict[key] = tf.cast(value, dtype)
                    continue
        if isinstance(value, dict):
            cast_input_dict[key] = cast_types(input_dict[key], dtype)
            continue
        if isinstance(value, list):
            cur_list = []
            for nest_value in value:
                if isinstance(nest_value, tf.Tensor):
                    if (
                        nest_value.dtype == tf.float16
                        or nest_value.dtype == tf.float32
                    ):
                        if nest_value.dtype.base_dtype != dtype.base_dtype:
                            cur_list.append(tf.cast(nest_value, dtype))
                            continue
                cur_list.append(nest_value)
            cast_input_dict[key] = cur_list
            continue
        cast_input_dict[key] = input_dict[key]
    return cast_input_dict


class Encoder:
    @staticmethod
    def get_required_params():
        return {}

    @staticmethod
    def get_optional_params():
        return {
            'regularizer': None,
            'regularizer_params': dict,
            'initializer': None,
            'initializer_params': dict,
            'dtype': [tf.float32, tf.float16, 'mixed'],
        }

    def __init__(self, params, model, name = 'encoder', mode = 'train'):
        check_params(
            params, self.get_required_params(), self.get_optional_params()
        )
        self._params = copy.deepcopy(params)
        self._model = model

        if 'dtype' not in self._params:
            if self._model:
                self._params['dtype'] = self._model.params['dtype']
            else:
                self._params['dtype'] = tf.float32

        self._name = name
        self._mode = mode
        self._compiled = False

    def encode(self, input_dict):
        if not self._compiled:
            if 'regularizer' not in self._params:
                if self._model and 'regularizer' in self._model.params:
                    self._params['regularizer'] = copy.deepcopy(
                        self._model.params['regularizer']
                    )
                    self._params['regularizer_params'] = copy.deepcopy(
                        self._model.params['regularizer_params']
                    )

            if 'regularizer' in self._params:
                init_dict = self._params.get('regularizer_params', {})
                if self._params['regularizer'] is not None:
                    self._params['regularizer'] = self._params['regularizer'](
                        **init_dict
                    )

            if self._params['dtype'] == 'mixed':
                self._params['dtype'] = tf.float16

        if 'initializer' in self.params:
            init_dict = self.params.get('initializer_params', {})
            initializer = self.params['initializer'](**init_dict)
        else:
            initializer = None

        self._compiled = True

        with tf.variable_scope(
            self._name, initializer = initializer, dtype = self.params['dtype']
        ):
            return self._encode(self._cast_types(input_dict))

    def _cast_types(self, input_dict):
        return cast_types(input_dict, self.params['dtype'])

    def _encode(self, input_dict):
        pass

    @property
    def params(self):
        return self._params

    @property
    def mode(self):
        return self._mode

    @property
    def name(self):
        return self._name


class TDNNEncoder(Encoder):
    @staticmethod
    def get_required_params():
        return dict(
            Encoder.get_required_params(),
            **{
                'dropout_keep_prob': float,
                'convnet_layers': list,
                'activation_fn': None,  # any valid callable
            }
        )

    @staticmethod
    def get_optional_params():
        return dict(
            Encoder.get_optional_params(),
            **{
                'data_format': ['channels_first', 'channels_last'],
                'normalization': [
                    None,
                    'batch_norm',
                    'layer_norm',
                    'instance_norm',
                ],
                'bn_momentum': float,
                'bn_epsilon': float,
                'use_conv_mask': bool,
                'drop_block_prob': float,
                'drop_block_index': int,
            }
        )

    def __init__(self, params, model, name = 'w2l_encoder', mode = 'train'):
        super(TDNNEncoder, self).__init__(params, model, name, mode)

    def _encode(self, input_dict):
        """Creates TensorFlow graph for Wav2Letter like encoder.

        Args:
        input_dict (dict): input dictionary that has to contain
            the following fields::
                input_dict = {
                "source_tensors": [
                    src_sequence (shape=[batch_size, sequence length, num features]),
                    src_length (shape=[batch_size])
                ]
                }

        Returns:
        dict: dictionary with the following tensors::

            {
            'outputs': hidden state, shape=[batch_size, sequence length, n_hidden]
            'src_length': tensor, shape=[batch_size]
            }
        """

        source_sequence, src_length = input_dict['source_tensors']

        num_pad = tf.constant(0)

        max_len = tf.reduce_max(src_length) + num_pad

        training = self._mode == 'train'
        dropout_keep_prob = (
            self.params['dropout_keep_prob'] if training else 1.0
        )
        regularizer = self.params.get('regularizer', None)
        data_format = self.params.get('data_format', 'channels_last')
        normalization = self.params.get('normalization', 'batch_norm')

        drop_block_prob = self.params.get('drop_block_prob', 0.0)
        drop_block_index = self.params.get('drop_block_index', -1)

        normalization_params = {}

        if self.params.get('use_conv_mask', False):
            mask = tf.sequence_mask(
                lengths = src_length,
                maxlen = max_len,
                dtype = source_sequence.dtype,
            )
            mask = tf.expand_dims(mask, 2)

        if normalization is None:
            conv_block = conv_actv
        elif normalization == 'batch_norm':
            conv_block = conv_bn_actv
            normalization_params['bn_momentum'] = self.params.get(
                'bn_momentum', 0.90
            )
            normalization_params['bn_epsilon'] = self.params.get(
                'bn_epsilon', 1e-3
            )
        elif normalization == 'layer_norm':
            conv_block = conv_ln_actv
        elif normalization == 'instance_norm':
            conv_block = conv_in_actv
        else:
            raise ValueError('Incorrect normalization')

        conv_inputs = source_sequence
        if data_format == 'channels_last':
            conv_feats = conv_inputs  # B T F
        else:
            conv_feats = tf.transpose(conv_inputs, [0, 2, 1])  # B F T

        residual_aggregation = []

        convnet_layers = self.params['convnet_layers']

        for idx_convnet in range(len(convnet_layers)):
            layer_type = convnet_layers[idx_convnet]['type']
            layer_repeat = convnet_layers[idx_convnet]['repeat']
            ch_out = convnet_layers[idx_convnet]['num_channels']
            kernel_size = convnet_layers[idx_convnet]['kernel_size']
            strides = convnet_layers[idx_convnet]['stride']
            padding = convnet_layers[idx_convnet]['padding']
            dilation = convnet_layers[idx_convnet]['dilation']
            dropout_keep = (
                convnet_layers[idx_convnet].get(
                    'dropout_keep_prob', dropout_keep_prob
                )
                if training
                else 1.0
            )
            residual = convnet_layers[idx_convnet].get('residual', False)
            residual_dense = convnet_layers[idx_convnet].get(
                'residual_dense', False
            )

            # For the first layer in the block, apply a mask
            if self.params.get('use_conv_mask', False):
                conv_feats = conv_feats * mask

            if residual:
                layer_res = conv_feats
                if residual_dense:
                    residual_aggregation.append(layer_res)
                    layer_res = residual_aggregation

            for idx_layer in range(layer_repeat):

                if padding == 'VALID':
                    src_length = (src_length - kernel_size[0]) // strides[0] + 1
                    max_len = (max_len - kernel_size[0]) // strides[0] + 1
                else:
                    src_length = (src_length + strides[0] - 1) // strides[0]
                    max_len = (max_len + strides[0] - 1) // strides[0]

                if idx_layer > 0 and self.params.get('use_conv_mask', False):
                    conv_feats = conv_feats * mask

                if self.params.get('use_conv_mask', False) and (
                    padding == 'VALID' or strides[0] > 1
                ):
                    mask = tf.sequence_mask(
                        lengths = src_length,
                        maxlen = max_len,
                        dtype = conv_feats.dtype,
                    )
                    mask = tf.expand_dims(mask, 2)

                if residual and idx_layer == layer_repeat - 1:
                    conv_feats = conv_bn_res_bn_actv(
                        layer_type = layer_type,
                        name = 'conv{}{}'.format(
                            idx_convnet + 1, idx_layer + 1
                        ),
                        inputs = conv_feats,
                        res_inputs = layer_res,
                        filters = ch_out,
                        kernel_size = kernel_size,
                        activation_fn = self.params['activation_fn'],
                        strides = strides,
                        padding = padding,
                        dilation = dilation,
                        regularizer = regularizer,
                        training = training,
                        data_format = data_format,
                        drop_block_prob = drop_block_prob,
                        drop_block = (drop_block_index == idx_convnet),
                        **normalization_params
                    )
                else:
                    conv_feats = conv_block(
                        layer_type = layer_type,
                        name = 'conv{}{}'.format(
                            idx_convnet + 1, idx_layer + 1
                        ),
                        inputs = conv_feats,
                        filters = ch_out,
                        kernel_size = kernel_size,
                        activation_fn = self.params['activation_fn'],
                        strides = strides,
                        padding = padding,
                        dilation = dilation,
                        regularizer = regularizer,
                        training = training,
                        data_format = data_format,
                        **normalization_params
                    )

                conv_feats = tf.nn.dropout(
                    x = conv_feats, keep_prob = dropout_keep
                )

        outputs = conv_feats

        if data_format == 'channels_first':
            outputs = tf.transpose(outputs, [0, 2, 1])

        return {'outputs': outputs, 'src_length': src_length}
