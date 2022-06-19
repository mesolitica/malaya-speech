# Copyright 2022 https://github.com/kssteven418/Squeezeformer
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

import tensorflow as tf
from ..utils import shape_list


class TimeReductionLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size=5,
        stride=2,
        dropout=0.0,
        name="time_reduction",
        **kwargs,
    ):
        super(TimeReductionLayer, self).__init__(name=name, **kwargs)
        self.stride = stride
        self.kernel_size = kernel_size
        dw_max = kernel_size ** -0.5
        pw_max = input_dim ** -0.5
        self.dw_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(kernel_size, 1), strides=self.stride,
            padding="valid", name=f"{name}_dw_conv",
            depth_multiplier=1,
            depthwise_initializer=tf.keras.initializers.RandomUniform(minval=-dw_max, maxval=dw_max),
            bias_initializer=tf.keras.initializers.RandomUniform(minval=-dw_max, maxval=dw_max),
        )
        #self.swish = tf.keras.layers.Activation(tf.nn.swish, name=f"{name}_swish_activation")
        self.pw_conv = tf.keras.layers.Conv2D(
            filters=output_dim, kernel_size=1, strides=1,
            padding="valid", name=f"{name}_pw_conv_2",
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-pw_max, maxval=pw_max),
            bias_initializer=tf.keras.initializers.RandomUniform(minval=-pw_max, maxval=pw_max),
        )

    def call(self, inputs, training=False, mask=None, pad_mask=None, **kwargs):
        B, T, E = shape_list(inputs)
        outputs = tf.reshape(inputs, [B, T, 1, E])
        _pad_mask = tf.expand_dims(tf.expand_dims(pad_mask, -1), -1)
        outputs = outputs * tf.cast(_pad_mask, "float32")
        padding = max(0, self.kernel_size - self.stride)
        outputs = tf.pad(outputs, [[0, 0], [0, padding], [0, 0], [0, 0]])
        outputs = self.dw_conv(outputs, training=training)
        outputs = self.pw_conv(outputs, training=training)
        B, T, _, E = shape_list(outputs)
        outputs = tf.reshape(outputs, [B, T, E])

        mask = mask[:, ::self.stride, ::self.stride]
        pad_mask = pad_mask[:, ::self.stride]
        _, L = shape_list(pad_mask)
        outputs = tf.pad(outputs, [[0, 0], [0, L - T], [0, 0]])

        return outputs, mask, pad_mask
