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
from ..utils import shape_list


class Embedding(tf.keras.layers.Layer):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        contraint = None,
        regularizer = None,
        initializer = None,
        **kwargs
    ):
        super(Embedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.contraint = tf.keras.constraints.get(contraint)
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.initializer = tf.keras.initializers.get(initializer)

    def build(self, input_shape):
        self.embeddings = tf.get_variable(
            'transducer/transducer_prediction/transducer_prediction_embedding/embeddings',
            [self.vocab_size, self.embed_dim],
            dtype = tf.float32,
            initializer = self.initializer,
        )
        self.built = True

    def call(self, inputs):
        outputs = tf.cast(tf.expand_dims(inputs, axis = -1), dtype = tf.int32)
        return tf.gather_nd(self.embeddings, outputs)

    def get_config(self):
        conf = super(Embedding, self).get_config()
        conf.update(
            {
                'vocab_size': self.vocab_size,
                'embed_dim': self.embed_dim,
                'contraint': self.contraint,
                'regularizer': self.regularizer,
                'initializer': self.initializer,
            }
        )
        return conf


def get_rnn(rnn_type):
    assert rnn_type in ['lstm', 'gru', 'rnn']

    if rnn_type == 'lstm':
        return tf.keras.layers.LSTM

    if rnn_type == 'gru':
        return tf.keras.layers.GRU

    return tf.keras.layers.SimpleRNN


def get_shape_invariants(tensor):
    shapes = shape_list(tensor)
    return tf.TensorShape([i if isinstance(i, int) else None for i in shapes])


def merge_two_last_dims(x):
    b, _, f, c = shape_list(x)
    return tf.reshape(x, shape = [b, -1, f * c])


class TimeReduction(tf.keras.layers.Layer):
    def __init__(self, factor: int, name: str = 'TimeReduction', **kwargs):
        super(TimeReduction, self).__init__(name = name, **kwargs)
        self.time_reduction_factor = factor

    def padding(self, time):
        new_time = (
            tf.math.ceil(time / self.time_reduction_factor)
            * self.time_reduction_factor
        )
        return tf.cast(new_time, dtype = tf.int32) - time

    def call(self, inputs, **kwargs):
        shape = shape_list(inputs)
        outputs = tf.pad(inputs, [[0, 0], [0, self.padding(shape[1])], [0, 0]])
        outputs = tf.reshape(
            outputs, [shape[0], -1, shape[-1] * self.time_reduction_factor]
        )
        return outputs

    def get_config(self):
        config = super(TimeReduction, self).get_config()
        config.update({'factor': self.time_reduction_factor})
        return config
