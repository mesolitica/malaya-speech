import tensorflow as tf
import numpy as np
from .layer import ACT2FN, get_initializer
from .model import (
    TFFastSpeechIntermediate,
    TFFastSpeechSelfOutput,
    TFFastSpeechOutput,
)


class TFFastSpeechSelfAttention(tf.keras.layers.Layer):
    """attention module for fastspeech."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                'The hidden size (%d) is not a multiple of the number of attention '
                'heads (%d)' % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.all_head_size = (
            self.num_attention_heads * config.attention_head_size
        )

        self.query = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer = get_initializer(config.initializer_range),
            name = 'query',
        )
        self.key = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer = get_initializer(config.initializer_range),
            name = 'key',
        )
        self.value = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer = get_initializer(config.initializer_range),
            name = 'value',
        )

        self.dropout = tf.keras.layers.Dropout(
            config.attention_probs_dropout_prob
        )
        self.config = config

    def transpose_for_scores(self, x, batch_size):
        """Transpose to calculate attention scores."""
        x = tf.reshape(
            x,
            (
                batch_size,
                -1,
                self.num_attention_heads,
                self.config.attention_head_size,
            ),
        )
        return tf.transpose(x, perm = [0, 2, 1, 3])

    def call(self, inputs, x, training = False):
        """Call logic."""
        hidden_states, attention_mask = inputs

        batch_size = tf.shape(hidden_states)[0]
        mixed_query_layer = self.query(x)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        attention_scores = tf.matmul(query_layer, key_layer, transpose_b = True)
        dk = tf.cast(
            tf.shape(key_layer)[-1], attention_scores.dtype
        )  # scale attention_scores
        attention_scores = attention_scores / tf.math.sqrt(dk)

        if attention_mask is not None:
            # extended_attention_masks for self attention encoder.
            extended_attention_mask = attention_mask[
                :, tf.newaxis, tf.newaxis, :
            ]
            extended_attention_mask = tf.cast(
                extended_attention_mask, attention_scores.dtype
            )
            extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
            attention_scores = attention_scores + extended_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(attention_scores, axis = -1)
        attention_probs = self.dropout(attention_probs, training = training)

        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, perm = [0, 2, 1, 3])
        context_layer = tf.reshape(
            context_layer, (batch_size, -1, self.all_head_size)
        )

        outputs = (
            (context_layer, attention_probs)
            if self.output_attentions
            else (context_layer,)
        )
        return outputs


class TFFastSpeechAttention(tf.keras.layers.Layer):
    """Fastspeech attention module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.self_attention = TFFastSpeechSelfAttention(config, name = 'self')
        self.dense_output = TFFastSpeechSelfOutput(config, name = 'output')

    def call(self, inputs, x, training = False):
        input_tensor, attention_mask = inputs

        self_outputs = self.self_attention(
            [input_tensor, attention_mask], x, training = training
        )
        attention_output = self.dense_output(
            [self_outputs[0], input_tensor], training = training
        )
        masked_attention_output = attention_output * tf.cast(
            tf.expand_dims(attention_mask, 2), dtype = attention_output.dtype
        )
        outputs = (masked_attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class TFFastSpeechLayer(tf.keras.layers.Layer):
    """Fastspeech module (FFT module on the paper)."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.attention = TFFastSpeechAttention(config, name = 'attention')
        self.intermediate = TFFastSpeechIntermediate(
            config, name = 'intermediate'
        )
        self.bert_output = TFFastSpeechOutput(config, name = 'output')

    def call(self, inputs, x, training = False):
        """Call logic."""
        hidden_states, attention_mask = inputs

        attention_outputs = self.attention(
            [hidden_states, attention_mask], x, training = training
        )
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(
            [attention_output, attention_mask], training = training
        )
        layer_output = self.bert_output(
            [intermediate_output, attention_output], training = training
        )
        masked_layer_output = layer_output * tf.cast(
            tf.expand_dims(attention_mask, 2), dtype = layer_output.dtype
        )
        outputs = (masked_layer_output,) + attention_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class TFFastSpeechDecoder(tf.keras.layers.Layer):
    """Fast Speech encoder module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = [
            TFFastSpeechLayer(config, name = 'layer_._{}'.format(i))
            for i in range(config.num_hidden_layers)
        ]

    def call(self, inputs, x, training = False):
        """Call logic."""
        hidden_states, attention_mask = inputs

        all_hidden_states = ()
        all_attentions = ()
        for _, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                [hidden_states, attention_mask], x, training = training
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs
