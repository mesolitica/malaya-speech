import tensorflow as tf
import numpy as np
from tensorflow.python.ops import init_ops_v2
from typing import List, Tuple
from ..utils import GroupNormalization, shape_list, _get_dtype

EPSILON = 1e-7


def gelu(x):
    cdf = 0.5 * (
        1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044_715 * tf.pow(x, 3))))
    )
    return x * cdf


def glu(x, dims=2):
    def conv(x):
        return tf.layers.conv1d(x, x.shape[-1], 1, padding='same')

    splitted = tf.split(x, 2, dims)
    return conv(splitted[0]) * tf.nn.sigmoid(conv(splitted[1]))


class VarianceScaling(
    init_ops_v2.VarianceScaling, tf.keras.initializers.Initializer
):
    def __call__(self, shape, dtype=None, **kwargs):
        return super(VarianceScaling, self).__call__(
            shape, dtype=_get_dtype(dtype), **kwargs
        )


class HeUniform(VarianceScaling):
    def __init__(self, seed=None):
        super(HeUniform, self).__init__(
            scale=2.0, mode='fan_in', distribution='uniform', seed=seed
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
            name='ConvFeatureExtractionModel', **kwargs
        )

        assert mode in {'default', 'layer_norm'}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = tf.keras.layers.Conv1D(
                    n_out,
                    k,
                    strides=stride,
                    use_bias=conv_bias,
                    kernel_initializer=HeUniform,
                )

                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, 'layer norm and group norm are exclusive'

            seq = tf.keras.Sequential()
            seq.add(make_conv())
            seq.add(tf.keras.layers.Dropout(dropout))
            if is_layer_norm:
                seq.add(tf.keras.layers.LayerNormalization())
            elif is_group_norm:
                seq.add(GroupNormalization(groups=dim))
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
                    is_layer_norm=mode == 'layer_norm',
                    is_group_norm=mode == 'default' and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def call(self, x, training=True):
        x = tf.expand_dims(x, -1)
        for conv in self.conv_layers:
            x = conv(x, training=training)
            x = gelu(x)
        return x
