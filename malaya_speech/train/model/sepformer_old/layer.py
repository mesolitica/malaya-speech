# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.keras import initializers
from tensorflow.python.keras import constraints
from tensorflow.python.keras import activations
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn
from tensorflow.python.keras.layers.convolutional import Conv1D


class Conv1DTranspose(Conv1D):
    def __init__(
        self,
        filters,
        kernel_size,
        strides = 1,
        padding = 'valid',
        output_padding = None,
        data_format = None,
        dilation_rate = 1,
        activation = None,
        use_bias = True,
        kernel_initializer = 'glorot_uniform',
        bias_initializer = 'zeros',
        kernel_regularizer = None,
        bias_regularizer = None,
        activity_regularizer = None,
        kernel_constraint = None,
        bias_constraint = None,
        **kwargs
    ):
        super(Conv1DTranspose, self).__init__(
            filters = filters,
            kernel_size = kernel_size,
            strides = strides,
            padding = padding,
            data_format = data_format,
            dilation_rate = dilation_rate,
            activation = activations.get(activation),
            use_bias = use_bias,
            kernel_initializer = initializers.get(kernel_initializer),
            bias_initializer = initializers.get(bias_initializer),
            kernel_regularizer = regularizers.get(kernel_regularizer),
            bias_regularizer = regularizers.get(bias_regularizer),
            activity_regularizer = regularizers.get(activity_regularizer),
            kernel_constraint = constraints.get(kernel_constraint),
            bias_constraint = constraints.get(bias_constraint),
            **kwargs
        )

        self.output_padding = output_padding
        if self.output_padding is not None:
            self.output_padding = conv_utils.normalize_tuple(
                self.output_padding, 1, 'output_padding'
            )
            for stride, out_pad in zip(self.strides, self.output_padding):
                if out_pad >= stride:
                    raise ValueError(
                        'Stride ' + str(self.strides) + ' must be '
                        'greater than output padding '
                        + str(self.output_padding)
                    )

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if len(input_shape) != 3:
            raise ValueError(
                'Inputs should have rank 3. Received input shape: '
                + str(input_shape)
            )
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError(
                'The channel dimension of the inputs '
                'should be defined. Found `None`.'
            )
        input_dim = int(input_shape[channel_axis])
        self.input_spec = InputSpec(ndim = 3, axes = {channel_axis: input_dim})
        kernel_shape = self.kernel_size + (self.filters, input_dim)

        self.kernel = self.add_weight(
            name = 'kernel',
            shape = kernel_shape,
            initializer = self.kernel_initializer,
            regularizer = self.kernel_regularizer,
            constraint = self.kernel_constraint,
            trainable = True,
            dtype = self.dtype,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name = 'bias',
                shape = (self.filters,),
                initializer = self.bias_initializer,
                regularizer = self.bias_regularizer,
                constraint = self.bias_constraint,
                trainable = True,
                dtype = self.dtype,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        inputs_shape = array_ops.shape(inputs)
        batch_size = inputs_shape[0]
        if self.data_format == 'channels_first':
            t_axis = 2
        else:
            t_axis = 1

        length = inputs_shape[t_axis]
        if self.output_padding is None:
            output_padding = None
        else:
            output_padding = self.output_padding[0]

        # Infer the dynamic output shape:
        out_length = conv_utils.deconv_output_length(
            length,
            self.kernel_size[0],
            padding = self.padding,
            output_padding = output_padding,
            stride = self.strides[0],
            dilation = self.dilation_rate[0],
        )
        if self.data_format == 'channels_first':
            output_shape = (batch_size, self.filters, out_length)
        else:
            output_shape = (batch_size, out_length, self.filters)
        data_format = conv_utils.convert_data_format(self.data_format, ndim = 3)

        output_shape_tensor = array_ops.stack(output_shape)
        outputs = nn_ops.conv1d_transpose(
            inputs,
            self.kernel,
            output_shape_tensor,
            strides = self.strides,
            padding = self.padding.upper(),
            data_format = data_format,
            dilations = self.dilation_rate,
        )

        if not context.executing_eagerly():
            # Infer the static output shape:
            out_shape = self.compute_output_shape(inputs.shape)
            outputs.set_shape(out_shape)

        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias, data_format = data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def _get_channel_axis(self):
        if self.data_format == 'channels_first':
            return -1 - self.rank
        else:
            return -1
