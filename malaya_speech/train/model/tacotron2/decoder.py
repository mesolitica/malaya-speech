import tensorflow as tf
from tensorflow.python.ops import control_flow_util
from tensorflow.contrib.seq2seq.python.ops.decoder import (
    Decoder,
    BaseDecoder,
    _transpose_batch_time,
)
from tensorflow.python.framework import ops
from tensorflow.python.eager import context
from tensorflow.python.ops import variable_scope


def _prepend_batch(batch_size, shape):
    if isinstance(batch_size, tf.Tensor):
        static_batch_size = tf.get_static_value(batch_size)
    else:
        static_batch_size = batch_size
    if static_batch_size is None:
        return tf.concat(([batch_size], shape), axis = 0)
    return [static_batch_size] + shape


def dynamic_decode(
    decoder,
    output_time_major: bool = False,
    impute_finished: bool = False,
    maximum_iterations = None,
    parallel_iterations: int = 32,
    swap_memory: bool = False,
    training = None,
    scope = None,
    **kwargs
):
    """Perform dynamic decoding with `decoder`.
    Calls initialize() once and step() repeatedly on the Decoder object.
    Args:
      decoder: A `Decoder` instance.
      output_time_major: Python boolean.  Default: `False` (batch major). If
        `True`, outputs are returned as time major tensors (this mode is
        faster). Otherwise, outputs are returned as batch major tensors (this
        adds extra time to the computation).
      impute_finished: Python boolean.  If `True`, then states for batch
        entries which are marked as finished get copied through and the
        corresponding outputs get zeroed out.  This causes some slowdown at
        each time step, but ensures that the final state and outputs have
        the correct values and that backprop ignores time steps that were
        marked as finished.
      maximum_iterations: A strictly positive `int32` scalar, the maximum
         allowed number of decoding steps. Default is `None` (decode until the
         decoder is fully done).
      parallel_iterations: Argument passed to `tf.while_loop`.
      swap_memory: Argument passed to `tf.while_loop`.
      training: Python boolean. Indicates whether the layer should behave
          in training  mode or in inference mode. Only relevant
          when `dropout` or `recurrent_dropout` is used.
      scope: Optional name scope to use.
      **kwargs: dict, other keyword arguments for dynamic_decode. It might
        contain arguments for `BaseDecoder` to initialize, which takes all
        tensor inputs during call().
    Returns:
      `(final_outputs, final_state, final_sequence_lengths)`.
    Raises:
      ValueError: if `maximum_iterations` is provided but is not a scalar.
    """
    with variable_scope.variable_scope(scope, 'decoder') as varscope:
        ctxt = ops.get_default_graph()._get_control_flow_context()
        is_xla = control_flow_util.GetContainingXLAContext(ctxt) is not None
        in_while_loop = (
            control_flow_util.GetContainingWhileContext(ctxt) is not None
        )

        if not context.executing_eagerly() and not in_while_loop:
            if varscope.caching_device is None:
                varscope.set_caching_device(lambda op: op.device)

        is_xla = not tf.executing_eagerly() and control_flow_util.GraphOrParentsInXlaContext(
            tf.compat.v1.get_default_graph()
        )

        if maximum_iterations is not None:
            maximum_iterations = tf.convert_to_tensor(
                maximum_iterations,
                dtype = tf.int32,
                name = 'maximum_iterations',
            )
            if maximum_iterations.shape.ndims != 0:
                raise ValueError('maximum_iterations must be a scalar')
            tf.debugging.assert_greater(
                maximum_iterations,
                0,
                message = 'maximum_iterations should be greater than 0',
            )
        elif is_xla:
            raise ValueError(
                'maximum_iterations is required for XLA compilation.'
            )

        if isinstance(decoder, Decoder):
            initial_finished, initial_inputs, initial_state = (
                decoder.initialize()
            )
        else:
            # For BaseDecoder that takes tensor inputs during call.
            decoder_init_input = kwargs.pop('decoder_init_input', None)
            decoder_init_kwargs = kwargs.pop('decoder_init_kwargs', {})
            initial_finished, initial_inputs, initial_state = decoder.initialize(
                decoder_init_input, **decoder_init_kwargs
            )

        zero_outputs = tf.nest.map_structure(
            lambda shape, dtype: tf.zeros(
                _prepend_batch(decoder.batch_size, shape), dtype = dtype
            ),
            decoder.output_size,
            decoder.output_dtype,
        )

        if maximum_iterations is not None:
            initial_finished = tf.logical_or(
                initial_finished, 0 >= maximum_iterations
            )
        initial_sequence_lengths = tf.zeros_like(
            initial_finished, dtype = tf.int32
        )
        initial_time = tf.constant(0, dtype = tf.int32)

        def _shape(batch_size, from_shape):
            if (
                not isinstance(from_shape, tf.TensorShape)
                or from_shape.ndims == 0
            ):
                return None
            else:
                batch_size = tf.get_static_value(
                    tf.convert_to_tensor(batch_size, name = 'batch_size')
                )
                return tf.TensorShape([batch_size]).concatenate(from_shape)

        dynamic_size = maximum_iterations is None or not is_xla
        # The dynamic shape `TensoArray` is not allowed in TFLite yet.
        dynamic_size = dynamic_size

        def _create_ta(s, d):
            return tf.TensorArray(
                dtype = d,
                size = 0 if dynamic_size else maximum_iterations,
                dynamic_size = dynamic_size,
                element_shape = _shape(decoder.batch_size, s),
            )

        initial_outputs_ta = tf.nest.map_structure(
            _create_ta, decoder.output_size, decoder.output_dtype
        )

        def condition(
            unused_time,
            unused_outputs_ta,
            unused_state,
            unused_inputs,
            finished,
            unused_sequence_lengths,
        ):
            return tf.logical_not(tf.reduce_all(finished))

        def body(time, outputs_ta, state, inputs, finished, sequence_lengths):
            """Internal while_loop body.
            Args:
              time: scalar int32 tensor.
              outputs_ta: structure of TensorArray.
              state: (structure of) state tensors and TensorArrays.
              inputs: (structure of) input tensors.
              finished: bool tensor (keeping track of what's finished).
              sequence_lengths: int32 tensor (keeping track of time of finish).
            Returns:
              `(time + 1, outputs_ta, next_state, next_inputs, next_finished,
                next_sequence_lengths)`.
              ```
            """
            (
                next_outputs,
                decoder_state,
                next_inputs,
                decoder_finished,
            ) = decoder.step(time, inputs, state, training)
            decoder_state_sequence_lengths = False
            if decoder.tracks_own_finished:
                next_finished = decoder_finished
                lengths = getattr(decoder_state, 'lengths', None)
                if lengths is not None:
                    # sequence lengths are provided by decoder_state.lengths;
                    # overwrite our sequence lengths.
                    decoder_state_sequence_lengths = True
                    sequence_lengths = tf.cast(lengths, tf.int32)
            else:
                next_finished = tf.logical_or(decoder_finished, finished)

            if decoder_state_sequence_lengths:
                # Just pass something through the loop; at the next iteration
                # we'll pull the sequence lengths from the decoder_state again.
                next_sequence_lengths = sequence_lengths
            else:
                next_sequence_lengths = tf.where(
                    tf.logical_not(finished),
                    tf.fill(tf.shape(sequence_lengths), time + 1),
                    sequence_lengths,
                )

            tf.nest.assert_same_structure(state, decoder_state)
            tf.nest.assert_same_structure(outputs_ta, next_outputs)
            tf.nest.assert_same_structure(inputs, next_inputs)

            # Zero out output values past finish
            if impute_finished:

                def zero_out_finished(out, zero):
                    if finished.shape.rank < zero.shape.rank:
                        broadcast_finished = tf.broadcast_to(
                            tf.expand_dims(finished, axis = -1), zero.shape
                        )
                        return tf.where(broadcast_finished, zero, out)
                    else:
                        return tf.where(finished, zero, out)

                emit = tf.nest.map_structure(
                    zero_out_finished, next_outputs, zero_outputs
                )
            else:
                emit = next_outputs

            # Copy through states past finish
            def _maybe_copy_state(new, cur):
                # TensorArrays and scalar states get passed through.
                if isinstance(cur, tf.TensorArray):
                    pass_through = True
                else:
                    new.set_shape(cur.shape)
                    pass_through = new.shape.ndims == 0
                if not pass_through:
                    broadcast_finished = tf.broadcast_to(
                        tf.expand_dims(finished, axis = -1), new.shape
                    )
                    return tf.where(broadcast_finished, cur, new)
                else:
                    return new

            if impute_finished:
                next_state = tf.nest.map_structure(
                    _maybe_copy_state, decoder_state, state
                )
            else:
                next_state = decoder_state

            outputs_ta = tf.nest.map_structure(
                lambda ta, out: ta.write(time, out), outputs_ta, emit
            )
            return (
                time + 1,
                outputs_ta,
                next_state,
                next_inputs,
                next_finished,
                next_sequence_lengths,
            )

        res = tf.while_loop(
            condition,
            body,
            loop_vars = (
                initial_time,
                initial_outputs_ta,
                initial_state,
                initial_inputs,
                initial_finished,
                initial_sequence_lengths,
            ),
            parallel_iterations = parallel_iterations,
            maximum_iterations = maximum_iterations,
            swap_memory = swap_memory,
        )

        final_outputs_ta = res[1]
        final_state = res[2]
        final_sequence_lengths = res[5]

        final_outputs = tf.nest.map_structure(
            lambda ta: ta.stack(), final_outputs_ta
        )

        try:
            final_outputs, final_state = decoder.finalize(
                final_outputs, final_state, final_sequence_lengths
            )
        except NotImplementedError:
            pass

        if not output_time_major:

            final_outputs = tf.nest.map_structure(
                _transpose_batch_time, final_outputs
            )

    return final_outputs, final_state, final_sequence_lengths
