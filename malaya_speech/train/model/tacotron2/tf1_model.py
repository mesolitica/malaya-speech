"""
MIT License

Copyright (c) 2018 Rayhane Mama, https://github.com/Rayhane-mamah/Tacotron-2/blob/master/LICENSE

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from .model import TFTacotronPrenet, TFTacotronPostnet
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.framework import ops, tensor_shape
from tensorflow.python.ops import (
    array_ops,
    check_ops,
    rnn_cell_impl,
    tensor_array_ops,
)
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import (
    BahdanauAttention,
)
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops, math_ops, nn_ops, variable_scope

_zero_state_tensors = rnn_cell_impl._zero_state_tensors


class TacotronEncoderCell(RNNCell):
    """Tacotron 2 Encoder Cell
	Passes inputs through a stack of convolutional layers then through a bidirectional LSTM
	layer to predict the hidden representation vector (or memory)
	"""

    def __init__(self, convolutional_layers, lstm_layer):
        """Initialize encoder parameters
		Args:
			convolutional_layers: Encoder convolutional block class
			lstm_layer: encoder bidirectional lstm layer class
		"""
        super(TacotronEncoderCell, self).__init__()
        # Initialize encoder layers
        self._convolutions = convolutional_layers
        self._cell = lstm_layer

    def __call__(self, inputs, input_lengths = None):
        # Pass input sequence through a stack of convolutional layers
        conv_output = self._convolutions(inputs)

        # Extract hidden representation from encoder lstm cells
        hidden_representation = self._cell(conv_output, input_lengths)

        # For shape visualization
        self.conv_output_shape = conv_output.shape
        return hidden_representation


class TacotronDecoderCellState(
    collections.namedtuple(
        'TacotronDecoderCellState',
        (
            'cell_state',
            'attention',
            'time',
            'alignments',
            'alignment_history',
            'max_attentions',
        ),
    )
):
    """`namedtuple` storing the state of a `TacotronDecoderCell`.
	Contains:
	  - `cell_state`: The state of the wrapped `RNNCell` at the previous time
		step.
	  - `attention`: The attention emitted at the previous time step.
	  - `time`: int32 scalar containing the current time step.
	  - `alignments`: A single or tuple of `Tensor`(s) containing the alignments
		 emitted at the previous time step for each attention mechanism.
	  - `alignment_history`: a single or tuple of `TensorArray`(s)
		 containing alignment matrices from all time steps for each attention
		 mechanism. Call `stack()` on each to convert to a `Tensor`.
	"""

    def replace(self, **kwargs):
        """Clones the current state while overwriting components provided by kwargs.
		"""
        return super(TacotronDecoderCellState, self)._replace(**kwargs)


class TacotronDecoderCell(RNNCell):
    def __init__(
        self,
        prenet,
        attention_mechanism,
        rnn_cell,
        frame_projection,
        stop_projection,
    ):
        """Initialize decoder parameters
		Args:
		    prenet: A tensorflow fully connected layer acting as the decoder pre-net
		    attention_mechanism: A _BaseAttentionMechanism instance, usefull to
			    learn encoder-decoder alignments
		    rnn_cell: Instance of RNNCell, main body of the decoder
		    frame_projection: tensorflow fully connected layer with r * num_mels output units
		    stop_projection: tensorflow fully connected layer, expected to project to a scalar
			    and through a sigmoid activation
			mask_finished: Boolean, Whether to mask decoder frames after the <stop_token>
		"""
        super(TacotronDecoderCell, self).__init__()
        # Initialize decoder layers
        self._prenet = prenet
        self._attention_mechanism = attention_mechanism
        self._cell = rnn_cell
        self._frame_projection = frame_projection
        self._stop_projection = stop_projection

        self._attention_layer_size = self._attention_mechanism.values.get_shape()[
            -1
        ].value

    def _batch_size_checks(self, batch_size, error_message):
        return [
            check_ops.assert_equal(
                batch_size,
                self._attention_mechanism.batch_size,
                message = error_message,
            )
        ]

    @property
    def output_size(self):
        return self._frame_projection.shape

    @property
    def state_size(self):
        """The `state_size` property of `TacotronDecoderCell`.
		Returns:
		  An `TacotronDecoderCell` tuple containing shapes used by this object.
		"""
        return TacotronDecoderCellState(
            cell_state = self._cell._cell.state_size,
            time = tensor_shape.TensorShape([]),
            attention = self._attention_layer_size,
            alignments = self._attention_mechanism.alignments_size,
            alignment_history = (),
            max_attentions = (),
        )

    def zero_state(self, batch_size, dtype):
        """Return an initial (zero) state tuple for this `AttentionWrapper`.
		Args:
		  batch_size: `0D` integer tensor: the batch size.
		  dtype: The internal state data type.
		Returns:
		  An `TacotronDecoderCellState` tuple containing zeroed out tensors and,
		  possibly, empty `TensorArray` objects.
		Raises:
		  ValueError: (or, possibly at runtime, InvalidArgument), if
			`batch_size` does not match the output size of the encoder passed
			to the wrapper object at initialization time.
		"""
        with ops.name_scope(
            type(self).__name__ + 'ZeroState', values = [batch_size]
        ):
            cell_state = self._cell._cell.zero_state(batch_size, dtype)
            error_message = (
                'When calling zero_state of TacotronDecoderCell %s: '
                % self._base_name
                + 'Non-matching batch sizes between the memory '
                '(encoder output) and the requested batch size.'
            )
            with ops.control_dependencies(
                self._batch_size_checks(batch_size, error_message)
            ):
                cell_state = nest.map_structure(
                    lambda s: array_ops.identity(
                        s, name = 'checked_cell_state'
                    ),
                    cell_state,
                )
            return TacotronDecoderCellState(
                cell_state = cell_state,
                time = array_ops.zeros([], dtype = tf.int32),
                attention = _zero_state_tensors(
                    self._attention_layer_size, batch_size, dtype
                ),
                alignments = self._attention_mechanism.initial_alignments(
                    batch_size, dtype
                ),
                alignment_history = tensor_array_ops.TensorArray(
                    dtype = dtype, size = 0, dynamic_size = True
                ),
                max_attentions = tf.zeros((batch_size,), dtype = tf.int32),
            )

    def __call__(self, inputs, state):
        # Information bottleneck (essential for learning attention)
        prenet_output = self._prenet(inputs)

        # Concat context vector and prenet output to form LSTM cells input (input feeding)
        LSTM_input = tf.concat([prenet_output, state.attention], axis = -1)

        # Unidirectional LSTM layers
        LSTM_output, next_cell_state = self._cell(LSTM_input, state.cell_state)

        # Compute the attention (context) vector and alignments using
        # the new decoder cell hidden state as query vector
        # and cumulative alignments to extract location features
        # The choice of the new cell hidden state (s_{i}) of the last
        # decoder RNN Cell is based on Luong et Al. (2015):
        # https://arxiv.org/pdf/1508.04025.pdf
        previous_alignments = state.alignments
        previous_alignment_history = state.alignment_history
        context_vector, alignments, cumulated_alignments, max_attentions = _compute_attention(
            self._attention_mechanism,
            LSTM_output,
            previous_alignments,
            attention_layer = None,
            prev_max_attentions = state.max_attentions,
        )

        # Concat LSTM outputs and context vector to form projections inputs
        projections_input = tf.concat([LSTM_output, context_vector], axis = -1)

        # Compute predicted frames and predicted <stop_token>
        cell_outputs = self._frame_projection(projections_input)
        stop_tokens = self._stop_projection(projections_input)

        # Save alignment history
        alignment_history = previous_alignment_history.write(
            state.time, alignments
        )

        # Prepare next decoder state
        next_state = TacotronDecoderCellState(
            time = state.time + 1,
            cell_state = next_cell_state,
            attention = context_vector,
            alignments = cumulated_alignments,
            alignment_history = alignment_history,
            max_attentions = max_attentions,
        )

        return (cell_outputs, stop_tokens), next_state


def _compute_attention(
    attention_mechanism,
    cell_output,
    attention_state,
    attention_layer,
    prev_max_attentions,
):
    """Computes the attention and alignments for a given attention_mechanism."""
    alignments, next_attention_state, max_attentions = attention_mechanism(
        cell_output,
        state = attention_state,
        prev_max_attentions = prev_max_attentions,
    )

    # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
    expanded_alignments = array_ops.expand_dims(alignments, 1)
    # Context is the inner product of alignments and values along the
    # memory time dimension.
    # alignments shape is
    #   [batch_size, 1, memory_time]
    # attention_mechanism.values shape is
    #   [batch_size, memory_time, memory_size]
    # the batched matmul is over memory_time, so the output shape is
    #   [batch_size, 1, memory_size].
    # we then squeeze out the singleton dim.
    context = math_ops.matmul(expanded_alignments, attention_mechanism.values)
    context = array_ops.squeeze(context, [1])

    if attention_layer is not None:
        attention = attention_layer(array_ops.concat([cell_output, context], 1))
    else:
        attention = context

    return attention, alignments, next_attention_state, max_attentions


def _location_sensitive_score(W_query, W_fil, W_keys):
    """Impelements Bahdanau-style (cumulative) scoring function.
	This attention is described in:
		J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
	  gio, “Attention-based models for speech recognition,” in Ad-
	  vances in Neural Information Processing Systems, 2015, pp.
	  577–585.
	#############################################################################
			  hybrid attention (content-based + location-based)
							   f = F * α_{i-1}
	   energy = dot(v_a, tanh(W_keys(h_enc) + W_query(h_dec) + W_fil(f) + b_a))
	#############################################################################
	Args:
		W_query: Tensor, shape '[batch_size, 1, attention_dim]' to compare to location features.
		W_location: processed previous alignments into location features, shape '[batch_size, max_time, attention_dim]'
		W_keys: Tensor, shape '[batch_size, max_time, attention_dim]', typically the encoder outputs.
	Returns:
		A '[batch_size, max_time]' attention score (energy)
	"""
    # Get the number of hidden units from the trailing dimension of keys
    dtype = W_query.dtype
    num_units = W_keys.shape[-1].value or array_ops.shape(W_keys)[-1]

    v_a = tf.get_variable(
        'attention_variable_projection',
        shape = [num_units],
        dtype = dtype,
        initializer = tf.contrib.layers.xavier_initializer(),
    )
    b_a = tf.get_variable(
        'attention_bias',
        shape = [num_units],
        dtype = dtype,
        initializer = tf.zeros_initializer(),
    )

    return tf.reduce_sum(v_a * tf.tanh(W_keys + W_query + W_fil + b_a), [2])


def _smoothing_normalization(e):
    """Applies a smoothing normalization function instead of softmax
	Introduced in:
		J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
	  gio, “Attention-based models for speech recognition,” in Ad-
	  vances in Neural Information Processing Systems, 2015, pp.
	  577–585.
	############################################################################
						Smoothing normalization function
				a_{i, j} = sigmoid(e_{i, j}) / sum_j(sigmoid(e_{i, j}))
	############################################################################
	Args:
		e: matrix [batch_size, max_time(memory_time)]: expected to be energy (score)
			values of an attention mechanism
	Returns:
		matrix [batch_size, max_time]: [0, 1] normalized alignments with possible
			attendance to multiple memory time steps.
	"""
    return tf.nn.sigmoid(e) / tf.reduce_sum(
        tf.nn.sigmoid(e), axis = -1, keepdims = True
    )


class LocationSensitiveAttention(BahdanauAttention):
    """Impelements Bahdanau-style (cumulative) scoring function.
	Usually referred to as "hybrid" attention (content-based + location-based)
	Extends the additive attention described in:
	"D. Bahdanau, K. Cho, and Y. Bengio, “Neural machine transla-
  tion by jointly learning to align and translate,” in Proceedings
  of ICLR, 2015."
	to use previous alignments as additional location features.
	This attention is described in:
	J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
  gio, “Attention-based models for speech recognition,” in Ad-
  vances in Neural Information Processing Systems, 2015, pp.
  577–585.
	"""

    def __init__(
        self,
        num_units,
        memory,
        hparams,
        is_training,
        mask_encoder = True,
        memory_sequence_length = None,
        smoothing = False,
        cumulate_weights = True,
        name = 'LocationSensitiveAttention',
    ):
        """Construct the Attention mechanism.
		Args:
			num_units: The depth of the query mechanism.
			memory: The memory to query; usually the output of an RNN encoder.  This
				tensor should be shaped `[batch_size, max_time, ...]`.
			mask_encoder (optional): Boolean, whether to mask encoder paddings.
			memory_sequence_length (optional): Sequence lengths for the batch entries
				in memory.  If provided, the memory tensor rows are masked with zeros
				for values past the respective sequence lengths. Only relevant if mask_encoder = True.
			smoothing (optional): Boolean. Determines which normalization function to use.
				Default normalization function (probablity_fn) is softmax. If smoothing is
				enabled, we replace softmax with:
						a_{i, j} = sigmoid(e_{i, j}) / sum_j(sigmoid(e_{i, j}))
				Introduced in:
					J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
				  gio, “Attention-based models for speech recognition,” in Ad-
				  vances in Neural Information Processing Systems, 2015, pp.
				  577–585.
				This is mainly used if the model wants to attend to multiple input parts
				at the same decoding step. We probably won't be using it since multiple sound
				frames may depend on the same character/phone, probably not the way around.
				Note:
					We still keep it implemented in case we want to test it. They used it in the
					paper in the context of speech recognition, where one phoneme may depend on
					multiple subsequent sound frames.
			name: Name to use when creating ops.
		"""
        # Create normalization function
        # Setting it to None defaults in using softmax
        normalization_function = (
            _smoothing_normalization if (smoothing == True) else None
        )
        memory_length = (
            memory_sequence_length if (mask_encoder == True) else None
        )
        super(LocationSensitiveAttention, self).__init__(
            num_units = num_units,
            memory = memory,
            memory_sequence_length = memory_length,
            probability_fn = normalization_function,
            name = name,
        )

        self.location_convolution = tf.layers.Conv1D(
            filters = hparams.attention_filters,
            kernel_size = hparams.attention_kernel,
            padding = 'same',
            use_bias = True,
            bias_initializer = tf.zeros_initializer(),
            name = 'location_features_convolution',
        )
        self.location_layer = tf.layers.Dense(
            units = num_units,
            use_bias = False,
            dtype = tf.float32,
            name = 'location_features_layer',
        )
        self._cumulate = cumulate_weights
        self.synthesis_constraint = (
            hparams.synthesis_constraint and not is_training
        )
        self.attention_win_size = tf.convert_to_tensor(
            hparams.attention_win_size, dtype = tf.int32
        )
        self.constraint_type = hparams.synthesis_constraint_type

    def __call__(self, query, state, prev_max_attentions):
        """Score the query based on the keys and values.
		Args:
			query: Tensor of dtype matching `self.values` and shape
				`[batch_size, query_depth]`.
			state (previous alignments): Tensor of dtype matching `self.values` and shape
				`[batch_size, alignments_size]`
				(`alignments_size` is memory's `max_time`).
		Returns:
			alignments: Tensor of dtype matching `self.values` and shape
				`[batch_size, alignments_size]` (`alignments_size` is memory's
				`max_time`).
		"""
        previous_alignments = state
        with variable_scope.variable_scope(
            None, 'Location_Sensitive_Attention', [query]
        ):

            # processed_query shape [batch_size, query_depth] -> [batch_size, attention_dim]
            processed_query = (
                self.query_layer(query) if self.query_layer else query
            )
            # -> [batch_size, 1, attention_dim]
            processed_query = tf.expand_dims(processed_query, 1)

            # processed_location_features shape [batch_size, max_time, attention dimension]
            # [batch_size, max_time] -> [batch_size, max_time, 1]
            expanded_alignments = tf.expand_dims(previous_alignments, axis = 2)
            # location features [batch_size, max_time, filters]
            f = self.location_convolution(expanded_alignments)
            # Projected location features [batch_size, max_time, attention_dim]
            processed_location_features = self.location_layer(f)

            # energy shape [batch_size, max_time]
            energy = _location_sensitive_score(
                processed_query, processed_location_features, self.keys
            )

        if self.synthesis_constraint:
            Tx = tf.shape(energy)[-1]
            # prev_max_attentions = tf.squeeze(prev_max_attentions, [-1])
            if self.constraint_type == 'monotonic':
                key_masks = tf.sequence_mask(prev_max_attentions, Tx)
                reverse_masks = tf.sequence_mask(
                    Tx - self.attention_win_size - prev_max_attentions, Tx
                )[:, ::-1]
            else:
                assert self.constraint_type == 'window'
                key_masks = tf.sequence_mask(
                    prev_max_attentions
                    - (
                        self.attention_win_size // 2
                        + (self.attention_win_size % 2 != 0)
                    ),
                    Tx,
                )
                reverse_masks = tf.sequence_mask(
                    Tx - (self.attention_win_size // 2) - prev_max_attentions,
                    Tx,
                )[:, ::-1]

            masks = tf.logical_or(key_masks, reverse_masks)
            paddings = tf.ones_like(energy) * (-2 ** 32 + 1)  # (N, Ty/r, Tx)
            energy = tf.where(tf.equal(masks, False), energy, paddings)

            # alignments shape = energy shape = [batch_size, max_time]
        alignments = self._probability_fn(energy, previous_alignments)
        max_attentions = tf.argmax(
            alignments, -1, output_type = tf.int32
        )  # (N, Ty/r)

        # Cumulate alignments
        if self._cumulate:
            next_state = alignments + previous_alignments
        else:
            next_state = alignments

        return alignments, next_state, max_attentions
