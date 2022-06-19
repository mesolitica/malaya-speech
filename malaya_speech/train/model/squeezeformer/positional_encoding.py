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

import math
import numpy as np
import tensorflow as tf
from ..utils import shape_list


class PositionalEncoding(tf.keras.layers.Layer):
    '''
    Same positional encoding method as NeMo library
    '''

    def __init__(self, d_model, max_len=5000, name="positional_encoding_nemo", **kwargs):
        super().__init__(trainable=False, name=name, **kwargs)
        self.max_len = max_len
        positions = tf.expand_dims(tf.range(self.max_len - 1, -max_len, -1.0, dtype=tf.float32), axis=1)
        pos_length = tf.shape(positions)[0]
        pe = np.zeros([pos_length, d_model], 'float32')
        div_term = np.exp(
            tf.range(0, d_model, 2, dtype=tf.float32) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = np.sin(positions * div_term)
        pe[:, 1::2] = np.cos(positions * div_term)
        pe = tf.convert_to_tensor(pe)
        self.pe = tf.expand_dims(pe, 0)

    def call(self, inputs, **kwargs):
        # inputs shape [B, T, V]
        _, length, dmodel = shape_list(inputs)
        center_pos = tf.shape(self.pe)[1] // 2
        start_pos = center_pos - length + 1
        end_pos = center_pos + length
        pos_emb = self.pe[:, start_pos:end_pos]
        return tf.cast(pos_emb, dtype=inputs.dtype)

    def get_config(self):
        conf = super().get_config()
        return conf.update({"max_len": self.max_len})
