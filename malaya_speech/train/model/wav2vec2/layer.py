import tensorflow as tf
import numpy as np
from tensorflow.python.ops import init_ops_v2
from typing import List, Tuple
from ..utils import GroupNormalization, shape_list, _get_dtype

EPSILON = 1e-7


def gumbel_distribution(input_shape):
    uniform_dist = tf.random.uniform(input_shape, 0, 1)
    gumbel_dist = -1 * tf.math.log(
        -1 * tf.math.log(uniform_dist + EPSILON) + EPSILON
    )

    return gumbel_dist


def gumbel_softmax(x, tau, axis = -1):
    x = x + gumbel_distribution(tf.shape(x))
    x = tf.nn.softmax(x / tau, axis = -1)
    return x


def gelu(x):
    cdf = 0.5 * (
        1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044_715 * tf.pow(x, 3))))
    )
    return x * cdf


def glu(x, dims = 2):
    def conv(x):
        return tf.layers.conv1d(x, x.shape[-1], 1, padding = 'same')

    splitted = tf.split(x, 2, dims)
    return conv(splitted[0]) * tf.nn.sigmoid(conv(splitted[1]))


class VarianceScaling(
    init_ops_v2.VarianceScaling, tf.keras.initializers.Initializer
):
    def __call__(self, shape, dtype = None, **kwargs):
        return super(VarianceScaling, self).__call__(
            shape, dtype = _get_dtype(dtype), **kwargs
        )


class HeUniform(VarianceScaling):
    def __init__(self, seed = None):
        super(HeUniform, self).__init__(
            scale = 2.0, mode = 'fan_in', distribution = 'uniform', seed = seed
        )

    def get_config(self):
        return {'seed': self.seed}


class ConvFeatureExtractionModel(tf.keras.layers.Layer):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = 'default',
        conv_bias: bool = False,
        **kwargs,
    ):
        super(ConvFeatureExtractionModel, self).__init__(
            name = 'ConvFeatureExtractionModel', **kwargs
        )

        assert mode in {'default', 'layer_norm'}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm = False,
            is_group_norm = False,
            conv_bias = False,
        ):
            def make_conv():
                conv = tf.keras.layers.Conv1D(
                    n_out,
                    k,
                    strides = stride,
                    use_bias = conv_bias,
                    kernel_initializer = HeUniform,
                )

                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, 'layer norm and group norm are exclusive'

            seq = tf.keras.Sequential()
            seq.add(make_conv())
            seq.add(tf.keras.layers.Dropout(dropout))
            if is_layer_norm:
                seq.add(tf.keras.layers.LayerNormalization)
            elif is_group_norm:
                seq.add(GroupNormalization(groups = dim))
            return seq

        in_d = 1
        self.conv_layers = []
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, 'invalid conv definition: ' + str(cl)
            (dim, k, stride) = cl
            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm = mode == 'layer_norm',
                    is_group_norm = mode == 'default' and i == 0,
                    conv_bias = conv_bias,
                )
            )
            in_d = dim

    def call(self, x, training = True):
        x = tf.expand_dims(x, -1)
        for conv in self.conv_layers:
            x = conv(x, training = training)
            x = gelu(x)
        return x


class GumbelVectorQuantizer(tf.keras.layers.Layer):
    def __init__(
        self,
        dim,
        num_vars,
        temp,
        groups,
        combine_groups,
        vq_dim,
        time_first = True,
        activation = gelu,
        weight_proj_depth = 1,
        weight_proj_factor = 1,
        **kwargs,
    ):
        super(GumbelVectorQuantizer, self).__init__(
            name = 'GumbelVectorQuantizer', **kwargs
        )
        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.num_vars = num_vars
        self.time_first = time_first

        assert (
            vq_dim % groups == 0
        ), f'dim {vq_dim} must be divisible by groups {groups} for concatenation'

        self.vq_dim = vq_dim
        var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1

        self.vars = tf.get_variable(
            name = 'vars',
            shape = [1, num_groups * num_vars, var_dim],
            initializer = tf.truncated_normal_initializer(),
        )
        if weight_proj_depth > 1:

            def block(input_dim, output_dim):
                return tf.keras.layers.Dense(
                    output_dim, activation = activation
                )

            inner_dim = self.input_dim * weight_proj_factor
            self.weight_proj = tf.keras.Sequential()
            for i in range(weight_proj_depth - 1):
                self.weight_proj.add(
                    block(self.input_dim if i == 0 else inner_dim, inner_dim)
                )
            self.weight_proj.add(tf.keras.layers.Dense(groups * num_vars))
        else:
            self.weight_proj = tf.keras.layers.Dense(groups * num_vars)

        if isinstance(temp, str):
            import ast

            temp = ast.literal_eval(temp)
        assert len(temp) == 3, f'{temp}, {len(temp)}'

        self.max_temp, self.min_temp, self.temp_decay = temp
        self.curr_temp = self.max_temp
        self.codebook_indices = None

    def call(self, x, produce_targets = False, training = True):
        result = {'num_vars': self.num_vars * self.groups}

        if not self.time_first:
            x = tf.transpose(x, [0, 2, 1])

        bsz, tsz, fsz = shape_list(x)
        x = tf.reshape(x, (-1, fsz))
        x = self.weight_proj(x)
        x = tf.reshape(x, (bsz * tsz * self.groups, -1))

        k = tf.argmax(x, axis = -1)
        hard_x = tf.one_hot(k, tf.shape(x)[-1])
        hard_x = tf.reshape(hard_x, (bsz * tsz, self.groups, -1))

        hard_probs = tf.reduce_mean(hard_x, axis = 0)

        result['code_perplexity'] = tf.reduce_sum(
            tf.exp(
                tf.reduce_sum(hard_probs * tf.log(hard_probs + 1e-7), axis = -1)
            )
        )
        avg_probs = tf.reduce_mean(
            tf.nn.softmax(
                tf.reshape(x, (bsz * tsz, self.groups, -1)), axis = -1
            ),
            axis = 0,
        )
        result['prob_perplexity'] = tf.reduce_sum(
            tf.exp(
                -tf.reduce_sum(avg_probs * tf.log(avg_probs + 1e-7), axis = -1)
            )
        )
        result['temp'] = self.curr_temp
        if training:
            x = gumbel_softmax(x, tau = self.curr_temp)
        else:
            x = hard_x

        x = tf.reshape(x, (bsz * tsz, -1))
        vars = self.vars
        if self.combine_groups:
            vars = tf.tile(vars, (1, self.groups, 1))

        if produce_targets:
            result['targets'] = tf.reshape(
                tf.argmax(
                    tf.reshape(x, (bsz * tsz * self.groups, -1)), axis = -1
                ),
                (bsz, tsz, self.groups),
            )

        x = tf.expand_dims(x, -1) * vars
        x = tf.reshape(x, (bsz * tsz, self.groups, self.num_vars, -1))
        x = tf.reduce_sum(x, axis = -2)
        x = tf.reshape(x, (bsz, tsz, -1))

        x.set_shape((None, None, self.vq_dim))

        if not self.time_first:
            x = tf.transpose(x, (0, 2, 1))

        result['x'] = x

        return result
