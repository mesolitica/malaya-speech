import numpy as np
import tensorflow as tf
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
from .config import Wav2Vec2Config
from ..utils import shape_list
from ..activation import get_tf_activation
from ..conv import Conv1D


def stable_softmax(logits: tf.Tensor, axis: Optional[int] = None, name: Optional[str] = None) -> tf.Tensor:
    """
    Stable wrapper that returns the same output as `tf.nn.softmax`, but that works reliably with XLA on CPU. It is
    meant as a workaround for the [following issue](https://github.com/tensorflow/tensorflow/issues/55682), and will be
    removed after it gets fixed. The arguments and outputs are the same as `tf.nn.softmax`, and relies on the fact that
    `softmax(x) = softmax(x + c)` (see https://ogunlao.github.io/2020/04/26/you_dont_really_know_softmax.html).
    Args:
        logits (`tf.Tensor`):
            Must be one of the following types: half, float32, float64.
        axis (`int`, *optional*):
            The dimension softmax would be performed on. The default is -1 which indicates the last dimension.
        name (`str`, *optional*):
            A name for the operation.
    Returns:
        `tf.Tensor`:
            A Tensor. Has the same type and shape as logits.
    """
    # TODO: When the issue linked above gets sorted, add a check on TF version here and use the original function if
    # it has the fix. After we drop the support for unfixed versions, remove this function.
    return tf.nn.softmax(logits=logits + 1e-9, axis=axis, name=name)


def get_initializer(initializer_range: float = 0.02):
    """
    Creates a `tf.initializers.TruncatedNormal` with the given range.
    Args:
        initializer_range (*float*, defaults to 0.02): Standard deviation of the initializer range.
    Returns:
        `tf.initializers.TruncatedNormal`: The truncated normal initializer.
    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


LARGE_NEGATIVE = -1e8
_HIDDEN_STATES_START_POSITION = 2


@dataclass
class TFCausalLMOutput:
    """
    Base class for causal language model (or autoregressive) outputs.
    Args:
        loss (`tf.Tensor` of shape `(n,)`, *optional*, where n is the number of non-masked labels, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[tf.Tensor] = None
    logits: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


class TFPreTrainedModel(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # Save config and origin of the pretrained weights if given in model
        self.config = config
        self.name_or_path = config.name_or_path


@dataclass
class TFWav2Vec2BaseModelOutput:
    """
    Output type of [`TFWav2Vec2BaseModelOutput`], with potential hidden states and attentions.
    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        extract_features (`tf.Tensor` of shape `(batch_size, sequence_length, conv_dim[-1])`):
            Sequence of extracted feature vectors of the last convolutional layer of the model.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: tf.Tensor = None
    extract_features: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


@dataclass
class TFBaseModelOutput:
    """
    Base class for model's outputs, with potential hidden states and attentions.
    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(tf.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


def _sample_without_replacement(distribution, num_samples):
    """
    Categorical sampling without replacement is currently not implemented. The gumbel-max trick will do for now - see
    https://github.com/tensorflow/tensorflow/issues/9260 for more info
    """
    z = -tf.math.log(tf.random.uniform(shape_list(distribution), 0, 1))
    _, indices = tf.nn.top_k(distribution + z, num_samples)
    return indices


def _scatter_values_on_batch_indices(values, batch_indices, output_shape):
    """
    Scatter function as in PyTorch with indices in format (batch_dim, indixes)
    """
    indices_shape = shape_list(batch_indices)
    # broadcast batch dim to indices_shape
    broad_casted_batch_dims = tf.reshape(
        tf.broadcast_to(tf.expand_dims(tf.range(indices_shape[0]), axis=-1), indices_shape), [1, -1]
    )
    # transform batch_indices to pair_indices
    pair_indices = tf.transpose(tf.concat([broad_casted_batch_dims, tf.reshape(batch_indices, [1, -1])], 0))
    # scatter values to pair indices
    return tf.scatter_nd(pair_indices, tf.reshape(values, [-1]), output_shape)


def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    min_masks: int = 0,
) -> tf.Tensor:
    """
    Computes random mask spans for a given shape
    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        attention_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob:
            probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_length: size of the mask
        min_masks: minimum number of masked spans
    Adapted from [fairseq's
    data_utils.py](https://github.com/pytorch/fairseq/blob/e0788f7007a8473a76db573985031f3c94201e79/fairseq/data/data_utils.py#L376).
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and"
            f" `sequence_length`: {sequence_length}`"
        )
    # compute number of masked spans in batch
    num_masked_spans = int(mask_prob * sequence_length / mask_length + tf.random.uniform((1,)))
    num_masked_spans = max(num_masked_spans, min_masks)

    # make sure num masked indices <= sequence_length
    if num_masked_spans * mask_length > sequence_length:
        num_masked_spans = sequence_length // mask_length

    # SpecAugment mask to fill
    spec_aug_mask = tf.zeros((batch_size, sequence_length), dtype=tf.int32)

    # uniform distribution to sample from, make sure that offset samples are < sequence_length
    uniform_dist = tf.ones((batch_size, sequence_length - (mask_length - 1)))

    # get random indices to mask
    spec_aug_mask_idxs = _sample_without_replacement(uniform_dist, num_masked_spans)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = tf.expand_dims(spec_aug_mask_idxs, -1)
    spec_aug_mask_idxs = tf.tile(spec_aug_mask_idxs, (1, 1, mask_length))
    spec_aug_mask_idxs = tf.reshape(spec_aug_mask_idxs, (batch_size, num_masked_spans * mask_length))

    offsets = tf.range(mask_length)[tf.newaxis, tf.newaxis, :]
    offsets = tf.tile(offsets, (batch_size, num_masked_spans, 1))
    offsets = tf.reshape(offsets, (batch_size, num_masked_spans * mask_length))

    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # scatter indices to mask
    spec_aug_mask = _scatter_values_on_batch_indices(
        tf.ones_like(spec_aug_mask_idxs), spec_aug_mask_idxs, spec_aug_mask.shape
    )

    return spec_aug_mask


def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None, past_key_values_length: int = 0):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    src_len = shape_list(mask)[1]
    tgt_len = tgt_len if tgt_len is not None else src_len
    one_cst = tf.constant(1.0)
    mask = tf.cast(mask, dtype=one_cst.dtype)
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

    return (one_cst - expanded_mask) * LARGE_NEGATIVE


class TFWav2Vec2GroupNorm(tf.keras.layers.Layer):
    """
    From tensorflow-addons https://www.tensorflow.org/addons/api_docs/python/tfa/layers/GroupNormalization
    """

    def __init__(
        self,
        groups: int = 32,
        axis: int = -1,
        epsilon: float = 1e-3,
        center: bool = True,
        scale: bool = True,
        beta_initializer: tf.keras.initializers.Initializer = "zeros",
        gamma_initializer: tf.keras.initializers.Initializer = "ones",
        beta_regularizer: tf.keras.regularizers.Regularizer = None,
        gamma_regularizer: tf.keras.regularizers.Regularizer = None,
        beta_constraint: tf.keras.constraints.Constraint = None,
        gamma_constraint: tf.keras.constraints.Constraint = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self._check_axis()

    def build(self, input_shape):

        self._check_if_input_shape_is_none(input_shape)
        self._set_number_of_groups_for_instance_norm(input_shape)
        self._check_size_of_dimensions(input_shape)
        self._create_input_spec(input_shape)

        self._add_gamma_weight(input_shape)
        self._add_beta_weight(input_shape)
        self.built = True
        super().build(input_shape)

    def call(self, inputs):

        input_shape = tf.keras.backend.int_shape(inputs)
        tensor_input_shape = tf.shape(inputs)

        reshaped_inputs, group_shape = self._reshape_into_groups(inputs, input_shape, tensor_input_shape)

        normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)

        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            outputs = tf.reshape(normalized_inputs, tensor_input_shape)
        else:
            outputs = normalized_inputs

        return outputs

    def get_config(self):
        config = {
            "groups": self.groups,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(self.gamma_initializer),
            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(self.gamma_regularizer),
            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape

    def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):

        group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            group_shape[self.axis] = input_shape[self.axis] // self.groups
            group_shape.insert(self.axis, self.groups)
            group_shape = tf.stack(group_shape)
            reshaped_inputs = tf.reshape(inputs, group_shape)
            return reshaped_inputs, group_shape
        else:
            return inputs, group_shape

    def _apply_normalization(self, reshaped_inputs, input_shape):

        group_shape = tf.keras.backend.int_shape(reshaped_inputs)
        group_reduction_axes = list(range(1, len(group_shape)))
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            axis = -2 if self.axis == -1 else self.axis - 1
        else:
            axis = -1 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)

        mean, variance = tf.nn.moments(reshaped_inputs, group_reduction_axes, keepdims=True)

        gamma, beta = self._get_reshaped_weights(input_shape)
        normalized_inputs = tf.nn.batch_normalization(
            reshaped_inputs,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon,
        )
        return normalized_inputs

    def _get_reshaped_weights(self, input_shape):
        broadcast_shape = self._create_broadcast_shape(input_shape)
        gamma = None
        beta = None
        if self.scale:
            gamma = tf.reshape(self.gamma, broadcast_shape)

        if self.center:
            beta = tf.reshape(self.beta, broadcast_shape)
        return gamma, beta

    def _check_if_input_shape_is_none(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError(
                "Axis "
                + str(self.axis)
                + " of input tensor should have a defined dimension but the layer received an input with shape "
                + str(input_shape)
                + "."
            )

    def _set_number_of_groups_for_instance_norm(self, input_shape):
        dim = input_shape[self.axis]

        if self.groups == -1:
            self.groups = dim

    def _check_size_of_dimensions(self, input_shape):

        dim = input_shape[self.axis]
        if dim < self.groups:
            raise ValueError(
                "Number of groups ("
                + str(self.groups)
                + ") cannot be more than the number of channels ("
                + str(dim)
                + ")."
            )

        if dim % self.groups != 0:
            raise ValueError(
                "Number of groups ("
                + str(self.groups)
                + ") must be a multiple of the number of channels ("
                + str(dim)
                + ")."
            )

    def _check_axis(self):

        if self.axis == 0:
            raise ValueError(
                "You are trying to normalize your batch axis. Do you want to use tf.layer.batch_normalization instead"
            )

    def _create_input_spec(self, input_shape):

        dim = input_shape[self.axis]
        self.input_spec = tf.keras.layers.InputSpec(ndim=len(input_shape), axes={self.axis: dim})

    def _add_gamma_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        else:
            self.gamma = None

    def _add_beta_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        else:
            self.beta = None

    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * len(input_shape)
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
            broadcast_shape.insert(self.axis, self.groups)
        else:
            broadcast_shape[self.axis] = self.groups
        return broadcast_shape


class TFWav2Vec2WeightNormConv1D(Conv1D):
    """Adapted from https://www.tensorflow.org/probability/api_docs/python/tfp/layers/weight_norm/WeightNorm"""

    def __init__(self, filters, kernel_size, groups, explicit_padding, **kwargs):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            groups=groups,
            padding="valid",
            use_bias=True,
            bias_initializer="he_normal",
            **kwargs,
        )
        self.explicit_padding = explicit_padding
        self.filter_axis = 2
        if tf.executing_eagerly():
            self.initialized = False
        else:
            self.initialized = True
        self.kernel_norm_axes = tf.constant([0, 1])

    def _init_norm(self):
        """Set the norm of the weight vector."""
        kernel_norm = tf.sqrt(tf.reduce_sum(tf.square(self.weight_v), axis=self.kernel_norm_axes))
        self.weight_g.assign(kernel_norm[:, tf.newaxis, tf.newaxis])

    def _normalize_kernel(self):
        """Generate normalized weights."""
        kernel = tf.nn.l2_normalize(self.weight_v, axis=self.kernel_norm_axes) * tf.transpose(self.weight_g)
        self.kernel = tf.transpose(kernel)

    def build(self, input_shape):
        if not self.built:
            # input_shape = input_shape.as_list()
            # Conv1D output shapes are checked at build time since TF 2.7, so we need to account for padding
            input_shape = tf.TensorShape((input_shape[0], input_shape[1] + self.explicit_padding * 2, input_shape[2]))
            super().build(input_shape)

            if tf.executing_eagerly():
                self.kernel = tf.Variable(tf.transpose(self.kernel), name="weight_v", trainable=True)
            else:
                self.kernel = tf.transpose(self.kernel)
            self.weight_v = self.kernel

            self.weight_g = self.add_weight(
                name="weight_g",
                shape=(int(self.weight_v.shape[self.filter_axis]), 1, 1),
                initializer="ones",
                dtype=self.weight_v.dtype,
                trainable=True,
            )
            if tf.executing_eagerly():
                self.bias = self.add_weight(name="bias", shape=(self.filters,), initializer="zeros", trainable=True)

    def call(self, inputs):
        if not self.initialized:
            self._init_norm()
            self.initialized = True

        self._normalize_kernel()

        padded_inputs = tf.pad(inputs, ((0, 0), (self.explicit_padding, self.explicit_padding), (0, 0)))
        output = super().call(padded_inputs)

        return output


class TFWav2Vec2NoLayerNormConvLayer(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, layer_id: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = tf.keras.layers.Conv1D(
            filters=self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            strides=config.conv_stride[layer_id],
            use_bias=config.conv_bias,
            name="conv",
        )
        self.activation = get_tf_activation(config.feat_extract_activation)

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class TFWav2Vec2LayerNormConvLayer(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, layer_id: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = tf.keras.layers.Conv1D(
            filters=self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            strides=config.conv_stride[layer_id],
            use_bias=config.conv_bias,
            name="conv",
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(name="layer_norm", epsilon=config.layer_norm_eps)
        self.activation = get_tf_activation(config.feat_extract_activation)

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class TFWav2Vec2GroupNormConvLayer(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, layer_id: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = tf.keras.layers.Conv1D(
            filters=self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            strides=config.conv_stride[layer_id],
            use_bias=config.conv_bias,
            name="conv",
        )
        self.activation = get_tf_activation(config.feat_extract_activation)
        self.layer_norm = TFWav2Vec2GroupNorm(
            groups=self.out_conv_dim, epsilon=config.layer_norm_eps, name="layer_norm"
        )

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class TFWav2Vec2PositionalConvEmbedding(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.conv = TFWav2Vec2WeightNormConv1D(
            filters=config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            groups=config.num_conv_pos_embedding_groups,
            explicit_padding=config.num_conv_pos_embeddings // 2,
            name="conv",
        )
        self.padding = TFWav2Vec2SamePadLayer(config.num_conv_pos_embeddings)
        self.activation = get_tf_activation(config.feat_extract_activation)

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class TFWav2Vec2SamePadLayer(tf.keras.layers.Layer):
    def __init__(self, num_conv_pos_embeddings, **kwargs):
        super().__init__(**kwargs)
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def call(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, : -self.num_pad_remove, :]
        return hidden_states


class TFWav2Vec2FeatureEncoder(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if config.feat_extract_norm == "group":
            conv_layers = [TFWav2Vec2GroupNormConvLayer(config, layer_id=0, name=f"conv_layers.{0}")] + [
                TFWav2Vec2NoLayerNormConvLayer(config, layer_id=i + 1, name=f"conv_layers.{i+1}")
                for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            conv_layers = [
                TFWav2Vec2LayerNormConvLayer(config, layer_id=i, name=f"conv_layers.{i}")
                for i in range(config.num_feat_extract_layers)
            ]
        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        self.conv_layers = conv_layers

    def call(self, input_values):
        hidden_states = tf.expand_dims(input_values, -1)
        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)
        return hidden_states


class TFWav2Vec2FeatureExtractor(TFWav2Vec2FeatureEncoder):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        warnings.warn(
            f"The class `{self.__class__.__name__}` has been depreciated "
            "and will be removed in Transformers v5. "
            f"Use `{self.__class__.__bases__[0].__name__}` instead.",
            FutureWarning,
        )


class TFWav2Vec2FeatureProjection(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        self.projection = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer="zeros",
            name="projection",
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.feat_proj_dropout)

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        return hidden_states, norm_hidden_states


# Copied from transformers.models.bart.modeling_tf_bart.TFBartAttention with TFBart->TFWav2Vec2
class TFWav2Vec2Attention(tf.keras.layers.Layer):
    """Multi-headed attention from "Attention Is All You Need"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")
        self.q_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")
        self.v_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")
        self.out_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")

    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))

    def call(
        self,
        hidden_states: tf.Tensor,
        key_value_states: Optional[tf.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[tf.Tensor]]] = None,
        attention_mask: Optional[tf.Tensor] = None,
        layer_head_mask: Optional[tf.Tensor] = None,
        training: Optional[bool] = False,
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = shape_list(hidden_states)

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = tf.concat([past_key_value[0], key_states], axis=2)
            value_states = tf.concat([past_key_value[1], value_states], axis=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(tf.Tensor, tf.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(tf.Tensor, tf.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = tf.reshape(self._shape(query_states, tgt_len, bsz), proj_shape)
        key_states = tf.reshape(key_states, proj_shape)
        value_states = tf.reshape(value_states, proj_shape)

        src_len = shape_list(key_states)[1]
        attn_weights = tf.matmul(query_states, key_states, transpose_b=True)

        # The tf.debugging asserts are not compliant with XLA then they
        # have to be disabled in other modes than eager.
        if tf.executing_eagerly():
            tf.debugging.assert_equal(
                shape_list(attn_weights),
                [bsz * self.num_heads, tgt_len, src_len],
                message=(
                    f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                    f" {shape_list(attn_weights)}"
                ),
            )

        if attention_mask is not None:
            # The tf.debugging asserts are not compliant with XLA then they
            # have to be disabled in other modes than eager.
            if tf.executing_eagerly():
                tf.debugging.assert_equal(
                    shape_list(attention_mask),
                    [bsz, 1, tgt_len, src_len],
                    message=(
                        f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                        f" {shape_list(attention_mask)}"
                    ),
                )

            attention_mask = tf.cast(attention_mask, dtype=attn_weights.dtype)
            attn_weights = tf.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len)) + attention_mask
            attn_weights = tf.reshape(attn_weights, (bsz * self.num_heads, tgt_len, src_len))

        attn_weights = stable_softmax(attn_weights, axis=-1)

        if layer_head_mask is not None:
            # The tf.debugging asserts are not compliant with XLA then they
            # have to be disabled in other modes than eager.
            if tf.executing_eagerly():
                tf.debugging.assert_equal(
                    shape_list(layer_head_mask),
                    [self.num_heads],
                    message=(
                        f"Head mask for a single layer should be of size {(self.num_heads)}, but is"
                        f" {shape_list(layer_head_mask)}"
                    ),
                )

            attn_weights = tf.reshape(layer_head_mask, (1, -1, 1, 1)) * tf.reshape(
                attn_weights, (bsz, self.num_heads, tgt_len, src_len)
            )
            attn_weights = tf.reshape(attn_weights, (bsz * self.num_heads, tgt_len, src_len))

        attn_probs = self.dropout(attn_weights, training=training)
        attn_output = tf.matmul(attn_probs, value_states)

        # The tf.debugging asserts are not compliant with XLA then they
        # have to be disabled in other modes than eager.
        if tf.executing_eagerly():
            tf.debugging.assert_equal(
                shape_list(attn_output),
                [bsz * self.num_heads, tgt_len, self.head_dim],
                message=(
                    f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                    f" {shape_list(attn_output)}"
                ),
            )

        attn_output = tf.transpose(
            tf.reshape(attn_output, (bsz, self.num_heads, tgt_len, self.head_dim)), (0, 2, 1, 3)
        )
        attn_output = tf.reshape(attn_output, (bsz, tgt_len, embed_dim))

        attn_output = self.out_proj(attn_output)
        attn_weights: tf.Tensor = tf.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len))

        return attn_output, attn_weights, past_key_value


class TFWav2Vec2FeedForward(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)

        self.intermediate_dropout = tf.keras.layers.Dropout(config.activation_dropout)

        self.intermediate_dense = tf.keras.layers.Dense(
            units=config.intermediate_size,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer="zeros",
            name="intermediate_dense",
        )
        self.intermediate_act_fn = get_tf_activation(config.hidden_act)

        self.output_dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer="zeros",
            name="output_dense",
        )
        self.output_dropout = tf.keras.layers.Dropout(config.hidden_dropout)

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states, training=training)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states, training=training)
        return hidden_states


class TFWav2Vec2EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)
        self.attention = TFWav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
            name="attention",
        )
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        self.feed_forward = TFWav2Vec2FeedForward(config, name="feed_forward")
        self.final_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="final_layer_norm"
        )

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        attn_residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, training=training
        )
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class TFWav2Vec2EncoderLayerStableLayerNorm(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)
        self.attention = TFWav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
            name="attention",
        )
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        self.feed_forward = TFWav2Vec2FeedForward(config, name="feed_forward")
        self.final_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="final_layer_norm"
        )

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, training=training
        )
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class TFWav2Vec2Encoder(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.pos_conv_embed = TFWav2Vec2PositionalConvEmbedding(config, name="pos_conv_embed")
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        self.layer = [TFWav2Vec2EncoderLayer(config, name=f"layers.{i}") for i in range(config.num_hidden_layers)]

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        training: Optional[bool] = False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            hidden_states = hidden_states * tf.expand_dims(attention_mask, -1)
            attention_mask = _expand_mask(attention_mask)
        else:
            attention_mask = None

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)
            if training and (dropout_probability < self.config.layerdrop):  # skip the layer
                continue

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return TFBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class TFWav2Vec2EncoderStableLayerNorm(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.pos_conv_embed = TFWav2Vec2PositionalConvEmbedding(config, name="pos_conv_embed")
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        self.layer = [
            TFWav2Vec2EncoderLayerStableLayerNorm(config, name=f"layers.{i}") for i in range(config.num_hidden_layers)
        ]

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        training: Optional[bool] = False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            hidden_states = hidden_states * tf.expand_dims(attention_mask, -1)
            attention_mask = _expand_mask(attention_mask)
        else:
            attention_mask = None

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states, training=training)

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)
            if training and (dropout_probability < self.config.layerdrop):  # skip the layer
                continue

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return TFBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class TFWav2Vec2MainLayer(tf.keras.layers.Layer):
    config_class = Wav2Vec2Config

    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.feature_extractor = TFWav2Vec2FeatureEncoder(config, name="feature_extractor")
        self.feature_projection = TFWav2Vec2FeatureProjection(config, name="feature_projection")

        if config.do_stable_layer_norm:
            self.encoder = TFWav2Vec2EncoderStableLayerNorm(config, name="encoder")
        else:
            self.encoder = TFWav2Vec2Encoder(config, name="encoder")

    def build(self, input_shape: tf.TensorShape):
        self.masked_spec_embed = self.add_weight(
            shape=(self.config.hidden_size,), initializer="uniform", trainable=True, name="masked_spec_embed"
        )

        super().build(input_shape)

    def _get_feat_extract_output_lengths(self, input_lengths: tf.Tensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    def _mask_hidden_states(self, hidden_states: tf.Tensor, mask_time_indices: Optional[tf.Tensor] = None):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """
        batch_size, sequence_length, hidden_size = shape_list(hidden_states)

        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            hidden_states = tf.where(
                tf.cast(mask_time_indices[:, :, tf.newaxis], tf.bool),
                self.masked_spec_embed[tf.newaxis, tf.newaxis, :],
                hidden_states,
            )

        elif self.config.mask_time_prob > 0:
            # generate indices & apply SpecAugment along time axis
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                min_masks=2,
            )
            hidden_states = tf.where(
                tf.cast(mask_time_indices[:, :, tf.newaxis], tf.bool),
                self.masked_spec_embed[tf.newaxis, tf.newaxis, :],
                hidden_states,
            )

        # apply SpecAugment along feature axis
        if self.config.mask_feature_prob > 0:
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
            )
            hidden_states = tf.where(mask_feature_indices[:, tf.newaxis, :], hidden_states, 0)

        return hidden_states

    def call(
        self,
        input_values: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        token_type_ids: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        head_mask: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        **kwargs: Any,
    ):

        extract_features = self.feature_extractor(
            tf.cast(input_values, tf.float32), training=training
        )
        # extract_features = tf.transpose(extract_features, perm=(0, 2, 1))

        if attention_mask is not None:
            # compute real output lengths according to convolution formula
            output_lengths = self._get_feat_extract_output_lengths(tf.reduce_sum(attention_mask, -1))

            attention_mask = tf.sequence_mask(
                output_lengths, maxlen=shape_list(extract_features)[1], dtype=extract_features.dtype
            )

        hidden_states, extract_features = self.feature_projection(extract_features, training=training)

        mask_time_indices = kwargs.get("mask_time_indices", None)
        if training:
            hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        hidden_states = encoder_outputs[0]

        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return TFWav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class TFWav2Vec2PreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Wav2Vec2Config
    base_model_prefix = "wav2vec2"
    main_input_name = "input_values"

    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        pad_token = 0.0
        input_values = tf.convert_to_tensor(np.random.rand(1, 16000), tf.float32)
        dummy_inputs = {
            "input_values": input_values,
            "attention_mask": tf.cast(tf.not_equal(input_values, pad_token), tf.float32),
        }
        return dummy_inputs

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)


class TFWav2Vec2Model(TFWav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.wav2vec2 = TFWav2Vec2MainLayer(config, name="wav2vec2")

    def call(
        self,
        input_values: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        token_type_ids: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        head_mask: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ):

        output_hidden_states = (
            output_hidden_states if output_hidden_states else self.config.output_hidden_states
        )
        output_attentions = (
            output_attentions if output_attentions else self.config.output_attentions
        )
        return_dict = return_dict if return_dict else self.config.return_dict

        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return outputs


class TFWav2Vec2ForCTC(TFWav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(config, **kwargs)

        self.wav2vec2 = TFWav2Vec2MainLayer(config, name="wav2vec2")
        self.dropout = tf.keras.layers.Dropout(config.final_dropout)
        self.lm_head = tf.keras.layers.Dense(config.vocab_size, name="lm_head")

    def call(
        self,
        input_values: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        token_type_ids: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        head_mask: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        labels: Optional[tf.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ):

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states, training=training)

        logits = self.lm_head(hidden_states)

        if labels is not None:

            if tf.reduce_max(labels) >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            attention_mask = (
                attention_mask
                if attention_mask is not None
                else tf.ones_like(input_values, dtype=tf.float32)
            )
            input_lengths = self.wav2vec2._get_feat_extract_output_lengths(tf.reduce_sum(attention_mask, axis=-1))

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = tf.cast(labels >= 0, tf.int32)
            target_lengths = tf.reduce_sum(labels_mask, axis=-1)

            loss = tf.nn.ctc_loss(
                logits=logits,
                labels=labels,
                logit_length=input_lengths,
                label_length=target_lengths,
                blank_index=self.config.pad_token_id,
                logits_time_major=False,
            )

            if self.config.ctc_loss_reduction == "sum":
                loss = tf.reduce_sum(loss)
            if self.config.ctc_loss_reduction == "mean":
                loss = tf.reduce_mean(loss)
        else:
            loss = None

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return TFCausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
