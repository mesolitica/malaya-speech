# Lint as: python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Keras-based attention layer."""

import tensorflow.compat.v1 as tf
# pylint: disable=g-classes-have-attributes

import collections
import math
import string

import numpy as np
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import backend
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.framework import ops
from .einsum_dense import EinsumDense


_CHR_IDX = string.ascii_lowercase


def maybe_init_scope(layer):
    """Open an `init_scope` if in V2 mode and using the keras graph.
    Args:
      layer: The Layer/Model that is currently active.
    Yields:
      None
    """
    # Don't open an init_scope in V1 mode or when using legacy tf.layers.
    if (ops.executing_eagerly_outside_functions() and
            getattr(layer, '_keras_style', True)):
        with ops.init_scope():
            yield
    else:
        yield


def _build_attention_equation(rank, attn_axes):
    """Builds einsum equations for the attention computation.

    Query, key, value inputs after projection are expected to have the shape as:
    `(bs, <non-attention dims>, <attention dims>, num_heads, channels)`.
    `bs` and `<non-attention dims>` are treated as `<batch dims>`.

    The attention operations can be generalized:
    (1) Query-key dot product:
    `(<batch dims>, <query attention dims>, num_heads, channels), (<batch dims>,
    <key attention dims>, num_heads, channels) -> (<batch dims>,
    num_heads, <query attention dims>, <key attention dims>)`
    (2) Combination:
    `(<batch dims>, num_heads, <query attention dims>, <key attention dims>),
    (<batch dims>, <value attention dims>, num_heads, channels) -> (<batch dims>,
    <query attention dims>, num_heads, channels)`

    Args:
      rank: Rank of query, key, value tensors.
      attn_axes: List/tuple of axes, `[-1, rank)`,
        that attention will be applied to.

    Returns:
      Einsum equations.
    """
    target_notation = _CHR_IDX[:rank]
    # `batch_dims` includes the head dim.
    batch_dims = tuple(np.delete(range(rank), attn_axes + (rank - 1,)))
    letter_offset = rank
    source_notation = ""
    for i in range(rank):
        if i in batch_dims or i == rank - 1:
            source_notation += target_notation[i]
        else:
            source_notation += _CHR_IDX[letter_offset]
            letter_offset += 1

    product_notation = "".join([target_notation[i] for i in batch_dims] +
                               [target_notation[i] for i in attn_axes] +
                               [source_notation[i] for i in attn_axes])
    dot_product_equation = "%s,%s->%s" % (source_notation, target_notation,
                                          product_notation)
    attn_scores_rank = len(product_notation)
    combine_equation = "%s,%s->%s" % (product_notation, source_notation,
                                      target_notation)
    return dot_product_equation, combine_equation, attn_scores_rank


def _build_proj_equation(free_dims, bound_dims, output_dims):
    """Builds an einsum equation for projections inside multi-head attention."""
    input_str = ""
    kernel_str = ""
    output_str = ""
    bias_axes = ""
    letter_offset = 0
    for i in range(free_dims):
        char = _CHR_IDX[i + letter_offset]
        input_str += char
        output_str += char

    letter_offset += free_dims
    for i in range(bound_dims):
        char = _CHR_IDX[i + letter_offset]
        input_str += char
        kernel_str += char

    letter_offset += bound_dims
    for i in range(output_dims):
        char = _CHR_IDX[i + letter_offset]
        kernel_str += char
        output_str += char
        bias_axes += char
    equation = "%s,%s->%s" % (input_str, kernel_str, output_str)

    return equation, bias_axes, len(output_str)


def _get_output_shape(output_rank, known_last_dims):
    return [None] * (output_rank - len(known_last_dims)) + list(known_last_dims)


def _large_compatible_negative(tensor_type):
    """Large negative number as Tensor.
    This function is necessary because the standard value for epsilon
    in this module (-1e9) cannot be represented using tf.float16
    Args:
      tensor_type: a dtype to determine the type.
    Returns:
      a large negative number.
    """
    if tensor_type == tf.float16:
        return tf.float16.min
    return -1e9


class Softmax(tf.keras.layers.Layer):
    """Softmax activation function.
    Example without mask:
    >>> inp = np.asarray([1., 2., 1.])
    >>> layer = tf.keras.layers.Softmax()
    >>> layer(inp).numpy()
    array([0.21194157, 0.5761169 , 0.21194157], dtype=float32)
    >>> mask = np.asarray([True, False, True], dtype=bool)
    >>> layer(inp, mask).numpy()
    array([0.5, 0. , 0.5], dtype=float32)
    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.
    Output shape:
      Same shape as the input.
    Args:
      axis: Integer, or list of Integers, axis along which the softmax
        normalization is applied.
    Call arguments:
      inputs: The inputs, or logits to the softmax layer.
      mask: A boolean mask of the same shape as `inputs`. Defaults to `None`. The
        mask specifies 1 to keep and 0 to mask.
    Returns:
      softmaxed output with the same shape as `inputs`.
    """

    def __init__(self, axis=-1, **kwargs):
        super(Softmax, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs, mask=None):
        if mask is not None:
            # Since mask is 1.0 for positions we want to keep and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -1e.9 for masked positions.
            adder = (1.0 - tf.cast(mask, inputs.dtype)) * (
                _large_compatible_negative(inputs.dtype))

            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            inputs += adder
        if isinstance(self.axis, (tuple, list)):
            if len(self.axis) > 1:
                return tf.exp(inputs - tf.reduce_logsumexp(
                    inputs, axis=self.axis, keepdims=True))
            else:
                return backend.softmax(inputs, axis=self.axis[0])


class MultiHeadAttention(tf.keras.layers.Layer):
    """MultiHeadAttention layer.

    This is an implementation of multi-headed attention as described in the paper
    "Attention is all you Need" (Vaswani et al., 2017).
    If `query`, `key,` `value` are the same, then
    this is self-attention. Each timestep in `query` attends to the
    corresponding sequence in `key`, and returns a fixed-width vector.

    This layer first projects `query`, `key` and `value`. These are
    (effectively) a list of tensors of length `num_attention_heads`, where the
    corresponding shapes are `(batch_size, <query dimensions>, key_dim)`,
    `(batch_size, <key/value dimensions>, key_dim)`,
    `(batch_size, <key/value dimensions>, value_dim)`.

    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor.

    Finally, the result tensor with the last dimension as value_dim can take an
    linear projection and return.

    When using MultiHeadAttention inside a custom Layer, the custom Layer must
    implement `build()` and call MultiHeadAttention's `_build_from_signature()`.
    This enables weights to be restored correctly when the model is loaded.
    TODO(b/172609172): link to documentation about calling custom build functions
    when used in a custom Layer.

    Examples:

    Performs 1D cross-attention over two sequence inputs with an attention mask.
    Returns the additional attention weights over heads.

    >>> layer = MultiHeadAttention(num_heads=2, key_dim=2)
    >>> target = tf.keras.Input(shape=[8, 16])
    >>> source = tf.keras.Input(shape=[4, 16])
    >>> output_tensor, weights = layer(target, source,
    ...                                return_attention_scores=True)
    >>> print(output_tensor.shape)
    (None, 8, 16)
    >>> print(weights.shape)
    (None, 2, 8, 4)

    Performs 2D self-attention over a 5D input tensor on axes 2 and 3.

    >>> layer = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(2, 3))
    >>> input_tensor = tf.keras.Input(shape=[5, 3, 4, 16])
    >>> output_tensor = layer(input_tensor, input_tensor)
    >>> print(output_tensor.shape)
    (None, 5, 3, 4, 16)

    Args:
      num_heads: Number of attention heads.
      key_dim: Size of each attention head for query and key.
      value_dim: Size of each attention head for value.
      dropout: Dropout probability.
      use_bias: Boolean, whether the dense layers use bias vectors/matrices.
      output_shape: The expected shape of an output tensor, besides the batch and
        sequence dims. If not specified, projects back to the key feature dim.
      attention_axes: axes over which the attention is applied. `None` means
        attention over all axes, but batch, heads, and features.
      kernel_initializer: Initializer for dense layer kernels.
      bias_initializer: Initializer for dense layer biases.
      kernel_regularizer: Regularizer for dense layer kernels.
      bias_regularizer: Regularizer for dense layer biases.
      activity_regularizer: Regularizer for dense layer activity.
      kernel_constraint: Constraint for dense layer kernels.
      bias_constraint: Constraint for dense layer kernels.

    Call arguments:
      query: Query `Tensor` of shape `(B, T, dim)`.
      value: Value `Tensor` of shape `(B, S, dim)`.
      key: Optional key `Tensor` of shape `(B, S, dim)`. If not given, will use
        `value` for both `key` and `value`, which is the most common case.
      attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
        attention to certain positions. The boolean mask specifies which query
        elements can attend to which key elements, 1 indicates attention and 0
        indicates no attention. Broadcasting can happen for the missing batch
        dimensions and the head dimension.
      return_attention_scores: A boolean to indicate whether the output should
        be `(attention_output, attention_scores)` if `True`, or `attention_output`
        if `False`. Defaults to `False`.
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (no dropout).
        Defaults to either using the training mode of the parent layer/model,
        or False (inference) if there is no parent layer.

    Returns:
      attention_output: The result of the computation, of shape `(B, T, E)`,
        where `T` is for target sequence shapes and `E` is the query input last
        dimension if `output_shape` is `None`. Otherwise, the multi-head outputs
        are project to the shape specified by `output_shape`.
      attention_scores: [Optional] multi-head attention coefficients over
        attention axes.
    """

    def __init__(self,
                 num_heads,
                 key_dim,
                 value_dim=None,
                 dropout=0.0,
                 use_bias=True,
                 output_shape=None,
                 attention_axes=None,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self._num_heads = num_heads
        self._key_dim = key_dim
        self._value_dim = value_dim if value_dim else key_dim
        self._dropout = dropout
        self._use_bias = use_bias
        self._output_shape = output_shape
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._bias_initializer = initializers.get(bias_initializer)
        self._kernel_regularizer = regularizers.get(kernel_regularizer)
        self._bias_regularizer = regularizers.get(bias_regularizer)
        self._activity_regularizer = regularizers.get(activity_regularizer)
        self._kernel_constraint = constraints.get(kernel_constraint)
        self._bias_constraint = constraints.get(bias_constraint)
        if attention_axes is not None and not isinstance(attention_axes,
                                                         collections.abc.Sized):
            self._attention_axes = (attention_axes,)
        else:
            self._attention_axes = attention_axes
        self._built_from_signature = False
        self._query_shape, self._key_shape, self._value_shape = None, None, None

    def get_config(self):
        config = {
            "num_heads": self._num_heads,
            "key_dim": self._key_dim,
            "value_dim": self._value_dim,
            "dropout": self._dropout,
            "use_bias": self._use_bias,
            "output_shape": self._output_shape,
            "attention_axes": self._attention_axes,
            "kernel_initializer":
                initializers.serialize(self._kernel_initializer),
            "bias_initializer":
                initializers.serialize(self._bias_initializer),
            "kernel_regularizer":
                regularizers.serialize(self._kernel_regularizer),
            "bias_regularizer":
                regularizers.serialize(self._bias_regularizer),
            "activity_regularizer":
                regularizers.serialize(self._activity_regularizer),
            "kernel_constraint":
                constraints.serialize(self._kernel_constraint),
            "bias_constraint":
                constraints.serialize(self._bias_constraint),
            "query_shape": self._query_shape,
            "key_shape": self._key_shape,
            "value_shape": self._value_shape,
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        # If the layer has a different build() function from the Keras default,
        # we need to trigger the customized build to create weights.
        query_shape = config.pop("query_shape")
        key_shape = config.pop("key_shape")
        value_shape = config.pop("value_shape")
        layer = cls(**config)
        if None in [query_shape, key_shape, value_shape]:
            logging.warning(
                "One of dimensions of the input shape is missing. It should have been"
                " memorized when the layer was serialized. "
                "%s is created without weights.",
                str(cls))
        else:
            layer._build_from_signature(query_shape, value_shape, key_shape)  # pylint: disable=protected-access
        return layer

    def _build_from_signature(self, query, value, key=None):
        """Builds layers and variables.

        Once the method is called, self._built_from_signature will be set to True.

        Args:
          query: Query tensor or TensorShape.
          value: Value tensor or TensorShape.
          key: Key tensor or TensorShape.
        """
        self._built_from_signature = True
        if hasattr(query, "shape"):
            self._query_shape = tf.TensorShape(query.shape)
        else:
            self._query_shape = tf.TensorShape(query)
        if hasattr(value, "shape"):
            self._value_shape = tf.TensorShape(value.shape)
        else:
            self._value_shape = tf.TensorShape(value)
        if key is None:
            self._key_shape = self._value_shape
        elif hasattr(key, "shape"):
            self._key_shape = tf.TensorShape(key.shape)
        else:
            self._key_shape = tf.TensorShape(key)

        common_kwargs = dict(
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint)
        # Any setup work performed only once should happen in an `init_scope`
        # to avoid creating symbolic Tensors that will later pollute any eager
        # operations.
        # with maybe_init_scope(self):
        with ops.init_scope():
            free_dims = self._query_shape.rank - 1
            einsum_equation, bias_axes, output_rank = _build_proj_equation(
                free_dims, bound_dims=1, output_dims=2)
            self._query_dense = EinsumDense(
                einsum_equation,
                output_shape=_get_output_shape(output_rank - 1,
                                               [self._num_heads, self._key_dim]),
                bias_axes=bias_axes if self._use_bias else None,
                name="query",
                **common_kwargs)
            einsum_equation, bias_axes, output_rank = _build_proj_equation(
                self._key_shape.rank - 1, bound_dims=1, output_dims=2)
            self._key_dense = EinsumDense(
                einsum_equation,
                output_shape=_get_output_shape(output_rank - 1,
                                               [self._num_heads, self._key_dim]),
                bias_axes=bias_axes if self._use_bias else None,
                name="key",
                **common_kwargs)
            einsum_equation, bias_axes, output_rank = _build_proj_equation(
                self._value_shape.rank - 1, bound_dims=1, output_dims=2)
            self._value_dense = EinsumDense(
                einsum_equation,
                output_shape=_get_output_shape(output_rank - 1,
                                               [self._num_heads, self._value_dim]),
                bias_axes=bias_axes if self._use_bias else None,
                name="value",
                **common_kwargs)

            # Builds the attention computations for multi-head dot product attention.
            # These computations could be wrapped into the keras attention layer once
            # it support mult-head einsum computations.
            self._build_attention(output_rank)
            self._output_dense = self._make_output_dense(
                free_dims, common_kwargs, "attention_output")

    def _make_output_dense(self, free_dims, common_kwargs, name=None):
        """Builds the output projection matrix.

        Args:
          free_dims: Number of free dimensions for einsum equation building.
          common_kwargs: Common keyword arguments for einsum layer.
          name: Name for the projection layer.

        Returns:
          Projection layer.
        """
        if self._output_shape:
            if not isinstance(self._output_shape, collections.abc.Sized):
                output_shape = [self._output_shape]
            else:
                output_shape = self._output_shape
        else:
            output_shape = [self._query_shape[-1]]
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            free_dims, bound_dims=2, output_dims=len(output_shape))
        return EinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(output_rank - 1, output_shape),
            bias_axes=bias_axes if self._use_bias else None,
            name=name,
            **common_kwargs)

    def _build_attention(self, rank):
        """Builds multi-head dot-product attention computations.

        This function builds attributes necessary for `_compute_attention` to
        costomize attention computation to replace the default dot-product
        attention.

        Args:
          rank: the rank of query, key, value tensors.
        """
        if self._attention_axes is None:
            self._attention_axes = tuple(range(1, rank - 2))
        else:
            self._attention_axes = tuple(self._attention_axes)
        self._dot_product_equation, self._combine_equation, attn_scores_rank = (
            _build_attention_equation(rank, attn_axes=self._attention_axes))
        norm_axes = tuple(
            range(attn_scores_rank - len(self._attention_axes), attn_scores_rank))
        self._softmax = Softmax(axis=norm_axes)
        self._dropout_layer = tf.keras.layers.Dropout(rate=self._dropout)

    def _masked_softmax(self, attention_scores, attention_mask=None):
        # Normalize the attention scores to probabilities.
        # `attention_scores` = [B, N, T, S]
        if attention_mask is not None:
            # The expand dim happens starting from the `num_heads` dimension,
            # (<batch_dims>, num_heads, <query_attention_dims, key_attention_dims>)
            mask_expansion_axis = -len(self._attention_axes) * 2 - 1
            for _ in range(len(attention_scores.shape) - len(attention_mask.shape)):
                attention_mask = tf.expand_dims(
                    attention_mask, axis=mask_expansion_axis)
        return self._softmax(attention_scores, attention_mask)

    def _compute_attention(self,
                           query,
                           key,
                           value,
                           attention_mask=None,
                           training=None):
        """Applies Dot-product attention with query, key, value tensors.

        This function defines the computation inside `call` with projected
        multi-head Q, K, V inputs. Users can override this function for customized
        attention implementation.

        Args:
          query: Projected query `Tensor` of shape `(B, T, N, key_dim)`.
          key: Projected key `Tensor` of shape `(B, T, N, key_dim)`.
          value: Projected value `Tensor` of shape `(B, T, N, value_dim)`.
          attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
            attention to certain positions.
          training: Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (doing nothing).

        Returns:
          attention_output: Multi-headed outputs of attention computation.
          attention_scores: Multi-headed attention weights.
        """
        # Note: Applying scalar multiply at the smaller end of einsum improves
        # XLA performance, but may introduce slight numeric differences in
        # the Transformer attention head.
        query = tf.multiply(query, 1.0 / math.sqrt(float(self._key_dim)))

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        attention_scores = tf.einsum(self._dot_product_equation, key,
                                     query)

        attention_scores = self._masked_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training)

        # `context_layer` = [B, T, N, H]
        attention_output = tf.einsum(self._combine_equation,
                                     attention_scores_dropout, value)
        return attention_output, attention_scores

    def call(self,
             query,
             value,
             key=None,
             attention_mask=None,
             return_attention_scores=False,
             training=None):
        if not self._built_from_signature:
            self._build_from_signature(query=query, value=value, key=key)
        if key is None:
            key = value

        #   N = `num_attention_heads`
        #   H = `size_per_head`
        # `query` = [B, T, N ,H]
        query = self._query_dense(query)

        # `key` = [B, S, N, H]
        key = self._key_dense(key)

        # `value` = [B, S, N, H]
        value = self._value_dense(value)

        attention_output, attention_scores = self._compute_attention(
            query, key, value, attention_mask, training)
        attention_output = self._output_dense(attention_output)

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output
