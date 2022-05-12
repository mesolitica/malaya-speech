# coding=utf-8
# Copyright 2022 The Pix2Seq Authors.
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
from ..utils import shape_list
from ..attention import MultiHeadAttention


def gelu(features, approximate=False):
    if approximate:
        coeff = tf.cast(0.044715, features.dtype)
        return 0.5 * features * (
            1.0 + tf.tanh(0.7978845608028654 *
                                (features + coeff * tf.pow(features, 3))))
    else:
        return 0.5 * features * (1.0 + tf.erf(
            features / tf.cast(1.4142135623730951, features.dtype)))


def suffix_id(i):
    """Return suffix id for layer/variable name."""
    return '' if i == 0 else '_%d' % i


def get_variable_initializer(name=None):
    if name is None:
        return tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02)


def get_angles(pos, i, dim):
    angle_rates = 1 / tf.pow(10000., tf.cast(2 * (i//2), tf.float32) / dim)
    return tf.cast(pos, tf.float32) * tf.cast(angle_rates, tf.float32)


def positional_encoding(coords, dim):
    """coords in (bsz, size), return (bsz, size, dim)."""
    angle_rads = get_angles(tf.expand_dims(coords, -1),
                            tf.range(dim)[tf.newaxis, tf.newaxis, :],
                            dim)

    # apply sin to even indices in the array; 2i
    angle_rads1 = tf.sin(angle_rads[:, :, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads2 = tf.cos(angle_rads[:, :, 1::2])

    pos_encoding = tf.concat([angle_rads1, angle_rads2], -1)

    return tf.cast(pos_encoding, dtype=tf.float32)


def get_2d_position_codes(height, width, out_dim, normalization_max=6.2831852):
    """Get 2d positional embedding with sin/cos codes.
    Args:
      height: a `int` specifying the height of the 2d image / feature map.
      width: a `int` specifying the width of the 2d image / feature map.
      out_dim: a `int` specifying the output dimension of the encoding.
        Must be divisible by 2.
      normalization_max: normalize coordinates between [0, normalization_max].
        If None, raw coordinates from 0 to height/width will be used.
    Returns:
      positional code of shape (1, height, width, out_dim)
    """
    y_coords = tf.cast(tf.range(height), tf.float32)
    if normalization_max is not None:
        y_coords = y_coords / (tf.cast(height, tf.float32) - 1) * normalization_max
    y_coords = positional_encoding(y_coords, out_dim//2)
    y_coords = tf.expand_dims(y_coords, 2)
    y_coords = tf.concat([y_coords, tf.zeros_like(y_coords)], -1)

    x_coords = tf.cast(tf.range(width), tf.float32)
    if normalization_max is not None:
        x_coords = x_coords / (tf.cast(width, tf.float32) - 1) * normalization_max
    x_coords = positional_encoding(x_coords, out_dim//2)
    x_coords = tf.expand_dims(x_coords, 1)
    x_coords = tf.concat([tf.zeros_like(x_coords), x_coords], -1)

    return y_coords + x_coords


def add_vis_pos_emb(self, pos_encoding, n_rows, n_cols, dim,
                    name_prefix=None, initializer=None):
    """Add vis_pos_emb variable/tensor to model instance referenced by `self`."""
    if name_prefix is None:
        name_prefix = self.name
    if initializer is None:
        initializer = get_variable_initializer()
    if pos_encoding == 'learned':
        self.vis_pos_emb = self.add_weight(
            shape=(n_rows * n_cols, dim), initializer=initializer,
            name='%s/vis_pos_embedding' % name_prefix)
    elif pos_encoding == 'sin_cos':
        sin_cos = get_2d_position_codes(
            n_rows, n_cols, dim, normalization_max=6.2831852)
        self.vis_pos_emb = tf.reshape(sin_cos, [n_rows * n_cols, dim])
    else:
        raise ValueError('Unknown pos encoding %s' % pos_encoding)


def add_cls_token_emb(self, dim, name_prefix=None, initializer=None):
    """Add cls_token_emb variable to model instance referenced by `self`."""
    if name_prefix is None:
        name_prefix = self.name
    if initializer is None:
        initializer = get_variable_initializer()
    self.cls_token_emb = self.add_weight(
        shape=(1, dim), initializer=initializer,
        name='%s/cls_token_embedding' % name_prefix)


class DropPath(tf.keras.layers.Layer):
    """For stochastic depth."""

    def __init__(self, drop_rate=0., **kwargs):
        """Initializes a drop path layer."""
        super(DropPath, self).__init__(**kwargs)
        self._drop_rate = drop_rate
        if self._drop_rate < 0 or self._drop_rate >= 1.0:
            raise ValueError('drop_rate {} is outside [0, 1)'.format(self._drop_rate))

    def call(self, x, training=False):
        """Performs a forward pass.
        Args:
          x: An input tensor of type tf.Tensor with shape [batch, height,
            width, channels].
          training: A boolean flag indicating whether training behavior should be
            used (default: False).
        Returns:
          The output tensor.
        """
        if self._drop_rate == 0. or not training:
            return x

        keep_rate = 1. - self._drop_rate
        xshape = tf.shape(x)
        drop_mask_shape = [xshape[0]] + [1] * (len(xshape) - 1)
        drop_mask = keep_rate + tf.random.uniform(drop_mask_shape, dtype=x.dtype)
        drop_mask = tf.math.divide(tf.floor(drop_mask), keep_rate)

        return x * drop_mask


class TransformerEncoderLayer(tf.keras.layers.Layer):

    def __init__(self,
                 dim,
                 mlp_ratio,
                 num_heads,
                 drop_path=0.1,
                 drop_units=0.1,
                 drop_att=0.,
                 **kwargs):
        super(TransformerEncoderLayer, self).__init__(**kwargs)
        self.mha_ln = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name='mha/ln')
        self.mha = MultiHeadAttention(
            num_heads, dim // num_heads, dropout=drop_att, name='mha')
        self.mlp = MLP(1, dim, mlp_ratio, drop_path, drop_units, name='mlp')
        self.dropp = DropPath(drop_path)

    def call(self, x, mask, training):
        # x shape (bsz, seq_len, dim_att), mask shape (bsz, seq_len, seq_len).
        x_ln = self.mha_ln(x)
        x_residual = self.mha(x_ln, x_ln, x_ln, mask, training=training)
        x = x + self.dropp(x_residual, training)
        x = self.mlp(x, training)
        return x


class FeedForwardLayer(tf.keras.layers.Layer):

    def __init__(self, dim_att, dim_mlp, drop_units=0.1, **kwargs):
        super(FeedForwardLayer, self).__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(
            dim_mlp, activation=gelu, name='dense1')
        self.dropout = tf.keras.layers.Dropout(drop_units)
        self.dense2 = tf.keras.layers.Dense(dim_att, name='dense2')

    def call(self, x, training):
        return self.dense2(self.dropout(self.dense1(x), training=training))


class MLP(tf.keras.layers.Layer):

    def __init__(self,
                 num_layers,
                 dim,
                 mlp_ratio,
                 drop_path=0.1,
                 drop_units=0.,
                 **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.mlp_layers = [
            FeedForwardLayer(dim, dim * mlp_ratio, drop_units,
                             name='ffn' + suffix_id(i))
            for i in range(num_layers)
        ]
        self.layernorms = [
            tf.keras.layers.LayerNormalization(
                epsilon=1e-6, name='ffn/ln' + suffix_id(i))
            for i in range(num_layers)
        ]
        self.dropp = DropPath(drop_path)

    def call(self, x, training, ret_list=False):
        x_list = [x]
        for i in range(self.num_layers):
            x_residual = self.mlp_layers[i](self.layernorms[i](x), training)
            x = x + self.dropp(x_residual, training)
            x_list.append(x)
        return (x, x_list) if ret_list else x


class TransformerEncoder(tf.keras.layers.Layer):

    def __init__(self,
                 num_layers,
                 dim,
                 mlp_ratio,
                 num_heads,
                 drop_path=0.1,
                 drop_units=0.1,
                 drop_att=0.,
                 **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.enc_layers = [
            TransformerEncoderLayer(
                dim, mlp_ratio, num_heads, drop_path, drop_units, drop_att,
                name='transformer_encoder' + suffix_id(i))
            for i in range(num_layers)
        ]

    def call(self, x, mask, training, ret_list=False):
        x_list = [x]
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask, training)
            x_list.append(x)
        return (x, x_list) if ret_list else x


class Model(tf.keras.Model):
    def __init__(self,
                 image_height,
                 image_width,
                 patch_size,
                 num_layers,
                 dim,
                 mlp_ratio,
                 num_heads,
                 drop_path=0.1,
                 drop_units=0.1,
                 drop_att=0.,
                 pos_encoding='learned',
                 use_cls_token=True,
                 **kwargs):
        super(Model, self).__init__(**kwargs)
        self.use_cls_token = use_cls_token
        self.patch_size = patch_size
        self.stem_conv = tf.keras.layers.Conv2D(
            filters=dim, kernel_size=patch_size, strides=patch_size,
            padding='VALID', use_bias=True, name='stem_conv')
        self.stem_ln = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name='stem_ln')
        if self.use_cls_token:
            add_cls_token_emb(self, dim)
        self.n_rows, self.n_cols = image_height//patch_size, image_width//patch_size
        add_vis_pos_emb(self, pos_encoding, self.n_rows, self.n_cols, dim)
        self.transformer_encoder = TransformerEncoder(
            num_layers, dim, mlp_ratio, num_heads, drop_path, drop_units, drop_att,
            name='transformer_encoder')
        self.output_ln = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name='ouput_ln')

    def call(self, images, training=True, ret_list=False):
        tokens = self.stem_conv(images)
        bsz, h, w, dim = shape_list(tokens)
        tokens = self.stem_ln(tf.reshape(tokens, [bsz, h * w, dim]))
        tokens = tokens + tf.expand_dims(self.vis_pos_emb, 0)
        if self.use_cls_token:
            cls_token = tf.tile(tf.expand_dims(self.cls_token_emb, 0), [bsz, 1, 1])
            tokens = tf.concat([cls_token, tokens], 1)

        tokens, x_list = self.transformer_encoder(
            tokens, None, training=training, ret_list=True)
        x = self.output_ln(tokens)
        return (x, x_list) if ret_list else x
