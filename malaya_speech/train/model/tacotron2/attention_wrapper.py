# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
from tensorflow.python.keras.engine import base_layer_utils


class AttentionMechanism(tf.keras.layers.Layer):
    """Base class for attention mechanisms.
    Common functionality includes:
      1. Storing the query and memory layers.
      2. Preprocessing and storing the memory.
    Note that this layer takes memory as its init parameter, which is an
    anti-pattern of Keras API, we have to keep the memory as init parameter for
    performance and dependency reason. Under the hood, during `__init__()`, it
    will invoke `base_layer.__call__(memory, setup_memory=True)`. This will let
    keras to keep track of the memory tensor as the input of this layer. Once
    the `__init__()` is done, then user can query the attention by
    `score = att_obj([query, state])`, and use it as a normal keras layer.
    Special attention is needed when adding using this class as the base layer
    for new attention:
      1. Build() could be invoked at least twice. So please make sure weights
         are not duplicated.
      2. Layer.get_weights() might return different set of weights if the
         instance has `query_layer`. The query_layer weights is not initialized
         until the memory is configured.
    Also note that this layer does not work with Keras model when
    `model.compile(run_eagerly=True)` due to the fact that this layer is
    stateful. The support for that will be added in a future version.
    """

    def __init__(
        self,
        memory,
        probability_fn: callable,
        query_layer = None,
        memory_layer = None,
        memory_sequence_length = None,
        **kwargs,
    ):
        """Construct base AttentionMechanism class.
        Args:
          memory: The memory to query; usually the output of an RNN encoder.
            This tensor should be shaped `[batch_size, max_time, ...]`.
          probability_fn: A `callable`. Converts the score and previous
            alignments to probabilities. Its signature should be:
            `probabilities = probability_fn(score, state)`.
          query_layer: Optional `tf.keras.layers.Layer` instance. The layer's
            depth must match the depth of `memory_layer`.  If `query_layer` is
            not provided, the shape of `query` must match that of
            `memory_layer`.
          memory_layer: Optional `tf.keras.layers.Layer` instance. The layer's
            depth must match the depth of `query_layer`.
            If `memory_layer` is not provided, the shape of `memory` must match
            that of `query_layer`.
          memory_sequence_length: (optional) Sequence lengths for the batch
            entries in memory. If provided, the memory tensor rows are masked
            with zeros for values past the respective sequence lengths.
          **kwargs: Dictionary that contains other common arguments for layer
            creation.
        """
        self.query_layer = query_layer
        self.memory_layer = memory_layer
        super().__init__(**kwargs)
        self.default_probability_fn = probability_fn
        self.probability_fn = probability_fn

        self.keys = None
        self.values = None
        self.batch_size = None
        self._memory_initialized = False
        self._check_inner_dims_defined = True
        self.supports_masking = True

        if memory is not None:
            # Setup the memory by self.__call__() with memory and
            # memory_seq_length. This will make the attention follow the keras
            # convention which takes all the tensor inputs via __call__().
            if memory_sequence_length is None:
                inputs = memory
            else:
                inputs = [memory, memory_sequence_length]

            self.values = super().__call__(inputs, setup_memory = True)

    @property
    def memory_initialized(self):
        """Returns `True` if this attention mechanism has been initialized with
        a memory."""
        return self._memory_initialized

    def build(self, input_shape):
        if not self._memory_initialized:
            # This is for setting up the memory, which contains memory and
            # optional memory_sequence_length. Build the memory_layer with
            # memory shape.
            if self.memory_layer is not None and not self.memory_layer.built:
                if isinstance(input_shape, list):
                    self.memory_layer.build(input_shape[0])
                else:
                    self.memory_layer.build(input_shape)
        else:
            # The input_shape should be query.shape and state.shape. Use the
            # query to init the query layer.
            if self.query_layer is not None and not self.query_layer.built:
                self.query_layer.build(input_shape[0])

    def __call__(self, inputs, **kwargs):
        """Preprocess the inputs before calling `base_layer.__call__()`.
        Note that there are situation here, one for setup memory, and one with
        actual query and state.
        1. When the memory has not been configured, we just pass all the param
           to `base_layer.__call__()`, which will then invoke `self.call()` with
           proper inputs, which allows this class to setup memory.
        2. When the memory has already been setup, the input should contain
           query and state, and optionally processed memory. If the processed
           memory is not included in the input, we will have to append it to
           the inputs and give it to the `base_layer.__call__()`. The processed
           memory is the output of first invocation of `self.__call__()`. If we
           don't add it here, then from keras perspective, the graph is
           disconnected since the output from previous call is never used.
        Args:
          inputs: the inputs tensors.
          **kwargs: dict, other keyeword arguments for the `__call__()`
        """
        # Allow manual memory reset
        if kwargs.get('setup_memory', False):
            self._memory_initialized = False

        if self._memory_initialized:
            if len(inputs) not in (2, 3):
                raise ValueError(
                    'Expect the inputs to have 2 or 3 tensors, got %d'
                    % len(inputs)
                )
            if len(inputs) == 2:
                # We append the calculated memory here so that the graph will be
                # connected.
                inputs.append(self.values)

        return super().__call__(inputs, **kwargs)

    def call(self, inputs, mask = None, setup_memory = False, **kwargs):
        """Setup the memory or query the attention.
        There are two case here, one for setup memory, and the second is query
        the attention score. `setup_memory` is the flag to indicate which mode
        it is. The input list will be treated differently based on that flag.
        Args:
          inputs: a list of tensor that could either be `query` and `state`, or
            `memory` and `memory_sequence_length`.
            `query` is the tensor of dtype matching `memory` and shape
            `[batch_size, query_depth]`.
            `state` is the tensor of dtype matching `memory` and shape
            `[batch_size, alignments_size]`. (`alignments_size` is memory's
            `max_time`).
            `memory` is the memory to query; usually the output of an RNN
            encoder. The tensor should be shaped `[batch_size, max_time, ...]`.
            `memory_sequence_length` (optional) is the sequence lengths for the
             batch entries in memory. If provided, the memory tensor rows are
            masked with zeros for values past the respective sequence lengths.
          mask: optional bool tensor with shape `[batch, max_time]` for the
            mask of memory. If it is not None, the corresponding item of the
            memory should be filtered out during calculation.
          setup_memory: boolean, whether the input is for setting up memory, or
            query attention.
          **kwargs: Dict, other keyword arguments for the call method.
        Returns:
          Either processed memory or attention score, based on `setup_memory`.
        """
        if setup_memory:
            if isinstance(inputs, list):
                if len(inputs) not in (1, 2):
                    raise ValueError(
                        'Expect inputs to have 1 or 2 tensors, got %d'
                        % len(inputs)
                    )
                memory = inputs[0]
                memory_sequence_length = inputs[1] if len(inputs) == 2 else None
                memory_mask = mask
            else:
                memory, memory_sequence_length = inputs, None
                memory_mask = mask
            self.setup_memory(memory, memory_sequence_length, memory_mask)
            # We force the self.built to false here since only memory is,
            # initialized but the real query/state has not been call() yet. The
            # layer should be build and call again.
            self.built = False
            # Return the processed memory in order to create the Keras
            # connectivity data for it.
            return self.values
        else:
            if not self._memory_initialized:
                raise ValueError(
                    'Cannot query the attention before the setup of memory'
                )
            if len(inputs) not in (2, 3):
                raise ValueError(
                    'Expect the inputs to have query, state, and optional '
                    'processed memory, got %d items' % len(inputs)
                )
            # Ignore the rest of the inputs and only care about the query and
            # state
            query, state = inputs[0], inputs[1]
            return self._calculate_attention(query, state)

    def setup_memory(
        self, memory, memory_sequence_length = None, memory_mask = None
    ):
        """Pre-process the memory before actually query the memory.
        This should only be called once at the first invocation of `call()`.
        Args:
          memory: The memory to query; usually the output of an RNN encoder.
            This tensor should be shaped `[batch_size, max_time, ...]`.
          memory_sequence_length (optional): Sequence lengths for the batch
            entries in memory. If provided, the memory tensor rows are masked
            with zeros for values past the respective sequence lengths.
          memory_mask: (Optional) The boolean tensor with shape `[batch_size,
            max_time]`. For any value equal to False, the corresponding value
            in memory should be ignored.
        """
        if memory_sequence_length is not None and memory_mask is not None:
            raise ValueError(
                'memory_sequence_length and memory_mask cannot be '
                'used at same time for attention.'
            )
        with tf.name_scope(self.name or 'BaseAttentionMechanismInit'):
            self.values = _prepare_memory(
                memory,
                memory_sequence_length = memory_sequence_length,
                memory_mask = memory_mask,
                check_inner_dims_defined = self._check_inner_dims_defined,
            )
            # Mark the value as check since the memory and memory mask might not
            # passed from __call__(), which does not have proper keras metadata.
            # TODO(omalleyt12): Remove this hack once the mask the has proper
            # keras history.
            base_layer_utils.mark_checked(self.values)
            if self.memory_layer is not None:
                self.keys = self.memory_layer(self.values)
            else:
                self.keys = self.values
            self.batch_size = self.keys.shape[0] or tf.shape(self.keys)[0]
            self._alignments_size = self.keys.shape[1] or tf.shape(self.keys)[1]
            if memory_mask is not None or memory_sequence_length is not None:
                unwrapped_probability_fn = self.default_probability_fn

                def _mask_probability_fn(score, prev):
                    return unwrapped_probability_fn(
                        _maybe_mask_score(
                            score,
                            memory_mask = memory_mask,
                            memory_sequence_length = memory_sequence_length,
                            score_mask_value = score.dtype.min,
                        ),
                        prev,
                    )

                self.probability_fn = _mask_probability_fn
        self._memory_initialized = True

    def _calculate_attention(self, query, state):
        raise NotImplementedError(
            '_calculate_attention need to be implemented by subclasses.'
        )

    def compute_mask(self, inputs, mask = None):
        # There real input of the attention is query and state, and the memory
        # layer mask shouldn't be pass down. Returning None for all output mask
        # here.
        return None, None

    def get_config(self):
        config = {}
        # Since the probability_fn is likely to be a wrapped function, the child
        # class should preserve the original function and how its wrapped.

        if self.query_layer is not None:
            config['query_layer'] = {
                'class_name': self.query_layer.__class__.__name__,
                'config': self.query_layer.get_config(),
            }
        if self.memory_layer is not None:
            config['memory_layer'] = {
                'class_name': self.memory_layer.__class__.__name__,
                'config': self.memory_layer.get_config(),
            }
        # memory is a required init parameter and its a tensor. It cannot be
        # serialized to config, so we put a placeholder for it.
        config['memory'] = None
        base_config = super().get_config()
        return {**base_config, **config}

    def _process_probability_fn(self, func_name):
        """Helper method to retrieve the probably function by string input."""
        valid_probability_fns = {'softmax': tf.nn.softmax}
        if func_name not in valid_probability_fns.keys():
            raise ValueError(
                'Invalid probability function: %s, options are %s'
                % (func_name, valid_probability_fns.keys())
            )
        return valid_probability_fns[func_name]

    @classmethod
    def deserialize_inner_layer_from_config(cls, config, custom_objects):
        """Helper method that reconstruct the query and memory from the config.
        In the get_config() method, the query and memory layer configs are
        serialized into dict for persistence, this method perform the reverse
        action to reconstruct the layer from the config.
        Args:
          config: dict, the configs that will be used to reconstruct the
            object.
          custom_objects: dict mapping class names (or function names) of
            custom (non-Keras) objects to class/functions.
        Returns:
          config: dict, the config with layer instance created, which is ready
            to be used as init parameters.
        """
        # Reconstruct the query and memory layer for parent class.
        # Instead of updating the input, create a copy and use that.
        config = config.copy()
        query_layer_config = config.pop('query_layer', None)
        if query_layer_config:
            query_layer = tf.keras.layers.deserialize(
                query_layer_config, custom_objects = custom_objects
            )
            config['query_layer'] = query_layer
        memory_layer_config = config.pop('memory_layer', None)
        if memory_layer_config:
            memory_layer = tf.keras.layers.deserialize(
                memory_layer_config, custom_objects = custom_objects
            )
            config['memory_layer'] = memory_layer
        return config

    @property
    def alignments_size(self):
        if isinstance(self._alignments_size, int):
            return self._alignments_size
        else:
            return tf.TensorShape([None])

    @property
    def state_size(self):
        return self.alignments_size

    def initial_alignments(self, batch_size, dtype):
        """Creates the initial alignment values for the `tfa.seq2seq.AttentionWrapper`
        class.
        This is important for attention mechanisms that use the previous
        alignment to calculate the alignment at the next time step
        (e.g. monotonic attention).
        The default behavior is to return a tensor of all zeros.
        Args:
          batch_size: `int32` scalar, the batch_size.
          dtype: The `dtype`.
        Returns:
          A `dtype` tensor shaped `[batch_size, alignments_size]`
          (`alignments_size` is the values' `max_time`).
        """
        return tf.zeros([batch_size, self._alignments_size], dtype = dtype)

    def initial_state(self, batch_size, dtype):
        """Creates the initial state values for the `tfa.seq2seq.AttentionWrapper` class.
        This is important for attention mechanisms that use the previous
        alignment to calculate the alignment at the next time step
        (e.g. monotonic attention).
        The default behavior is to return the same output as
        `initial_alignments`.
        Args:
          batch_size: `int32` scalar, the batch_size.
          dtype: The `dtype`.
        Returns:
          A structure of all-zero tensors with shapes as described by
          `state_size`.
        """
        return self.initial_alignments(batch_size, dtype)


def _bahdanau_score(
    processed_query, keys, attention_v, attention_g = None, attention_b = None
):
    """Implements Bahdanau-style (additive) scoring function.
    This attention has two forms.  The first is Bhandanau attention,
    as described in:
    Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
    "Neural Machine Translation by Jointly Learning to Align and Translate."
    ICLR 2015. https://arxiv.org/abs/1409.0473
    The second is the normalized form.  This form is inspired by the
    weight normalization article:
    Tim Salimans, Diederik P. Kingma.
    "Weight Normalization: A Simple Reparameterization to Accelerate
     Training of Deep Neural Networks."
    https://arxiv.org/abs/1602.07868
    To enable the second form, set please pass in attention_g and attention_b.
    Args:
      processed_query: Tensor, shape `[batch_size, num_units]` to compare to
        keys.
      keys: Processed memory, shape `[batch_size, max_time, num_units]`.
      attention_v: Tensor, shape `[num_units]`.
      attention_g: Optional scalar tensor for normalization.
      attention_b: Optional tensor with shape `[num_units]` for normalization.
    Returns:
      A `[batch_size, max_time]` tensor of unnormalized score values.
    """
    # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
    processed_query = tf.expand_dims(processed_query, 1)
    if attention_g is not None and attention_b is not None:
        normed_v = (
            attention_g
            * attention_v
            * tf.math.rsqrt(tf.reduce_sum(tf.square(attention_v)))
        )
        return tf.reduce_sum(
            normed_v * tf.tanh(keys + processed_query + attention_b), [2]
        )
    else:
        return tf.reduce_sum(attention_v * tf.tanh(keys + processed_query), [2])


class BahdanauAttention(AttentionMechanism):
    """Implements Bahdanau-style (additive) attention.
    This attention has two forms.  The first is Bahdanau attention,
    as described in:
    Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
    "Neural Machine Translation by Jointly Learning to Align and Translate."
    ICLR 2015. https://arxiv.org/abs/1409.0473
    The second is the normalized form.  This form is inspired by the
    weight normalization article:
    Tim Salimans, Diederik P. Kingma.
    "Weight Normalization: A Simple Reparameterization to Accelerate
     Training of Deep Neural Networks."
    https://arxiv.org/abs/1602.07868
    To enable the second form, construct the object with parameter
    `normalize=True`.
    """

    def __init__(
        self,
        units,
        memory = None,
        memory_sequence_length = None,
        normalize: bool = False,
        probability_fn: str = 'softmax',
        kernel_initializer = 'glorot_uniform',
        dtype = None,
        name: str = 'BahdanauAttention',
        **kwargs,
    ):
        """Construct the Attention mechanism.
        Args:
          units: The depth of the query mechanism.
          memory: The memory to query; usually the output of an RNN encoder.
            This tensor should be shaped `[batch_size, max_time, ...]`.
          memory_sequence_length: (optional): Sequence lengths for the batch
            entries in memory.  If provided, the memory tensor rows are masked
            with zeros for values past the respective sequence lengths.
          normalize: Python boolean.  Whether to normalize the energy term.
          probability_fn: (optional) string, the name of function to convert
            the attention score to probabilities. The default is `softmax`
            which is `tf.nn.softmax`. Other options is `hardmax`, which is
            hardmax() within this module. Any other value will result into
            validation error. Default to use `softmax`.
          kernel_initializer: (optional), the name of the initializer for the
            attention kernel.
          dtype: The data type for the query and memory layers of the attention
            mechanism.
          name: Name to use when creating ops.
          **kwargs: Dictionary that contains other common arguments for layer
            creation.
        """
        self.probability_fn_name = probability_fn
        probability_fn = self._process_probability_fn(self.probability_fn_name)

        def wrapped_probability_fn(score, _):
            return probability_fn(score)

        query_layer = kwargs.pop('query_layer', None)
        if not query_layer:
            query_layer = tf.keras.layers.Dense(
                units, name = 'query_layer', use_bias = False, dtype = dtype
            )
        memory_layer = kwargs.pop('memory_layer', None)
        if not memory_layer:
            memory_layer = tf.keras.layers.Dense(
                units, name = 'memory_layer', use_bias = False, dtype = dtype
            )
        self.units = units
        self.normalize = normalize
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.attention_v = None
        self.attention_g = None
        self.attention_b = None
        super().__init__(
            memory = memory,
            memory_sequence_length = memory_sequence_length,
            query_layer = query_layer,
            memory_layer = memory_layer,
            probability_fn = wrapped_probability_fn,
            name = name,
            dtype = dtype,
            **kwargs,
        )

    def build(self, input_shape):
        super().build(input_shape)
        if self.attention_v is None:
            self.attention_v = tf.get_variable(
                'attention_v',
                [self.units],
                dtype = self.dtype,
                initializer = self.kernel_initializer,
            )
        if (
            self.normalize
            and self.attention_g is None
            and self.attention_b is None
        ):
            self.attention_g = tf.get_variable(
                'attention_g',
                initializer = tf.constant_initializer(
                    math.sqrt(1.0 / self.units)
                ),
                shape = (),
            )
            self.attention_b = tf.get_variable(
                'attention_b',
                shape = [self.units],
                initializer = tf.zeros_initializer(),
            )
        self.built = True

    def _calculate_attention(self, query, state):
        """Score the query based on the keys and values.
        Args:
          query: Tensor of dtype matching `self.values` and shape
            `[batch_size, query_depth]`.
          state: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]`
            (`alignments_size` is memory's `max_time`).
        Returns:
          alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]` (`alignments_size` is memory's
            `max_time`).
          next_state: same as alignments.
        """
        processed_query = self.query_layer(query) if self.query_layer else query
        score = _bahdanau_score(
            processed_query,
            self.keys,
            self.attention_v,
            attention_g = self.attention_g,
            attention_b = self.attention_b,
        )
        alignments = self.probability_fn(score, state)
        next_state = alignments
        return alignments, next_state

    def get_config(self):
        # yapf: disable
        config = {
            "units": self.units,
            "normalize": self.normalize,
            "probability_fn": self.probability_fn_name,
            "kernel_initializer": tf.keras.initializers.serialize(
                self.kernel_initializer)
        }
        # yapf: enable

        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects = None):
        config = AttentionMechanism.deserialize_inner_layer_from_config(
            config, custom_objects = custom_objects
        )
        return cls(**config)


def _prepare_memory(
    memory,
    memory_sequence_length = None,
    memory_mask = None,
    check_inner_dims_defined = True,
):
    """Convert to tensor and possibly mask `memory`.
    Args:
      memory: `Tensor`, shaped `[batch_size, max_time, ...]`.
      memory_sequence_length: `int32` `Tensor`, shaped `[batch_size]`.
      memory_mask: `boolean` tensor with shape [batch_size, max_time]. The
        memory should be skipped when the corresponding mask is False.
      check_inner_dims_defined: Python boolean.  If `True`, the `memory`
        argument's shape is checked to ensure all but the two outermost
        dimensions are fully defined.
    Returns:
      A (possibly masked), checked, new `memory`.
    Raises:
      ValueError: If `check_inner_dims_defined` is `True` and not
        `memory.shape[2:].is_fully_defined()`.
    """
    memory = tf.nest.map_structure(
        lambda m: tf.convert_to_tensor(m, name = 'memory'), memory
    )
    if memory_sequence_length is not None and memory_mask is not None:
        raise ValueError(
            "memory_sequence_length and memory_mask can't be provided at same time."
        )
    if memory_sequence_length is not None:
        memory_sequence_length = tf.convert_to_tensor(
            memory_sequence_length, name = 'memory_sequence_length'
        )
    if check_inner_dims_defined:

        def _check_dims(m):
            if not m.shape[2:].is_fully_defined():
                raise ValueError(
                    'Expected memory %s to have fully defined inner dims, '
                    'but saw shape: %s' % (m.name, m.shape)
                )

        tf.nest.map_structure(_check_dims, memory)
    if memory_sequence_length is None and memory_mask is None:
        return memory
    elif memory_sequence_length is not None:
        seq_len_mask = tf.sequence_mask(
            memory_sequence_length,
            maxlen = tf.shape(tf.nest.flatten(memory)[0])[1],
            dtype = tf.nest.flatten(memory)[0].dtype,
        )
    else:
        # For memory_mask is not None
        seq_len_mask = tf.cast(
            memory_mask, dtype = tf.nest.flatten(memory)[0].dtype
        )

    def _maybe_mask(m, seq_len_mask):
        """Mask the memory based on the memory mask."""
        rank = m.shape.ndims
        rank = rank if rank is not None else tf.rank(m)
        extra_ones = tf.ones(rank - 2, dtype = tf.int32)
        seq_len_mask = tf.reshape(
            seq_len_mask, tf.concat((tf.shape(seq_len_mask), extra_ones), 0)
        )
        return m * seq_len_mask

    return tf.nest.map_structure(lambda m: _maybe_mask(m, seq_len_mask), memory)


def _maybe_mask_score(
    score,
    memory_sequence_length = None,
    memory_mask = None,
    score_mask_value = None,
):
    """Mask the attention score based on the masks."""
    if memory_sequence_length is None and memory_mask is None:
        return score
    if memory_sequence_length is not None and memory_mask is not None:
        raise ValueError(
            "memory_sequence_length and memory_mask can't be provided at same time."
        )
    if memory_sequence_length is not None:
        message = 'All values in memory_sequence_length must greater than zero.'
        with tf.control_dependencies(
            [
                tf.debugging.assert_positive(  # pylint: disable=bad-continuation
                    memory_sequence_length, message = message
                )
            ]
        ):
            memory_mask = tf.sequence_mask(
                memory_sequence_length, maxlen = tf.shape(score)[1]
            )
    score_mask_values = score_mask_value * tf.ones_like(score)
    return tf.where(memory_mask, score, score_mask_values)
