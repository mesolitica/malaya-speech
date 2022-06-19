# Copyright 2020 Huy Le Nguyen (@usimarit)
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
from ..utils import shape_list, merge_two_last_dims

logger = tf.get_logger()


class Conv2dSubsampling(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        strides: int = 2,
        kernel_size: int = 3,
        ds: bool = False,
        name="Conv2dSubsampling",
        **kwargs,
    ):
        super(Conv2dSubsampling, self).__init__(name=name, **kwargs)
        self.strides = strides
        self.kernel_size = kernel_size
        assert self.strides == 2 and self.kernel_size == 3  # Fix this for simplicity
        conv1_max = kernel_size ** -1
        conv2_max = (kernel_size ** 2 * filters) ** -0.5
        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size,
            strides=strides, padding="valid", name=f"{name}_1",
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-conv1_max, maxval=conv1_max),
            bias_initializer=tf.keras.initializers.RandomUniform(minval=-conv1_max, maxval=conv1_max),
        )
        self.ds = ds
        if not ds:
            logger.info("Subsampling with full conv")
            self.conv2 = tf.keras.layers.Conv2D(
                filters=filters, kernel_size=kernel_size,
                strides=strides, padding="valid", name=f"{name}_2",
                kernel_initializer=tf.keras.initializers.RandomUniform(minval=-conv2_max, maxval=conv2_max),
                bias_initializer=tf.keras.initializers.RandomUniform(minval=-conv2_max, maxval=conv2_max),
            )
            self.time_reduction_factor = self.conv1.strides[0] + self.conv2.strides[0]
        else:
            logger.info("Subsampling with DS conv")
            dw_max = (kernel_size ** 2) ** -0.5
            pw_max = filters ** -0.5
            self.dw_conv = tf.keras.layers.DepthwiseConv2D(
                kernel_size=(kernel_size, kernel_size), strides=strides,
                padding="valid", name=f"{name}_2_dw",
                depth_multiplier=1,
                depthwise_initializer=tf.keras.initializers.RandomUniform(minval=-dw_max, maxval=dw_max),
                bias_initializer=tf.keras.initializers.RandomUniform(minval=-dw_max, maxval=dw_max),
            )
            self.pw_conv = tf.keras.layers.Conv2D(
                filters=filters, kernel_size=1, strides=1,
                padding="valid", name=f"{name}_2_pw",
                kernel_initializer=tf.keras.initializers.RandomUniform(minval=-pw_max, maxval=pw_max),
                bias_initializer=tf.keras.initializers.RandomUniform(minval=-pw_max, maxval=pw_max),
            )
            self.time_reduction_factor = self.conv1.strides[0] + self.dw_conv.strides[0]

    def call(self, inputs, training=False, **kwargs):
        _, L, H, _ = shape_list(inputs)
        assert H == 80
        outputs = tf.pad(inputs, [[0, 0], [0, 1], [0, 1], [0, 0]])
        outputs = self.conv1(outputs, training=training)
        outputs = tf.nn.relu(outputs)
        outputs = tf.pad(outputs, [[0, 0], [0, 1], [0, 1], [0, 0]])
        if not self.ds:
            outputs = self.conv2(outputs, training=training)
        else:
            outputs = self.dw_conv(outputs, training=training)
            outputs = self.pw_conv(outputs, training=training)
        outputs = tf.nn.relu(outputs)
        _, L, H, _ = shape_list(outputs)
        assert H == 20
        return merge_two_last_dims(outputs)

    def get_config(self):
        conf = super(Conv2dSubsampling, self).get_config()
        conf.update(self.conv1.get_config())
        conf.update(self.conv2.get_config())
        return conf
