import tensorflow as tf

INF = 1.0 * 1e7
EOS_ID = 1


def _expand_to_beam_size(tensor, beam_size):
    """Tiles a given tensor by beam_size.
  Args:
    tensor: tensor to tile [batch_size, ...]
    beam_size: How much to tile the tensor by.
  Returns:
    Tiled tensor [batch_size, beam_size, ...]
  """
    tensor = tf.expand_dims(tensor, axis = 1)
    tile_dims = [1] * tensor.shape.ndims
    tile_dims[1] = beam_size

    return tf.tile(tensor, tile_dims)


def get_state_shape_invariants(tensor):
    """Returns the shape of the tensor but sets middle dims to None."""
    shape = tensor.shape.as_list()
    for i in range(1, len(shape) - 1):
        shape[i] = None
    return tf.TensorShape(shape)


def compute_batch_indices(batch_size, beam_size):
    """Computes the i'th coordinate that contains the batch index for gathers.
  Batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..]. It says which
  batch the beam item is in. This will create the i of the i,j coordinate
  needed for the gather.
  Args:
    batch_size: Batch size
    beam_size: Size of the beam.
  Returns:
    batch_pos: [batch_size, beam_size] tensor of ids
  """
    batch_pos = tf.range(batch_size * beam_size) // beam_size
    batch_pos = tf.reshape(batch_pos, [batch_size, beam_size])
    return batch_pos


def beam_search(
    symbols_to_logits_fn,
    initial_ids,
    beam_size,
    vocab_size,
    alpha,
    states = None,
    eos_id = EOS_ID,
    stop_early = True,
    use_top_k_with_unique = True,
):
    batch_size = tf.shape(initial_ids)[0]
    initial_log_probs = tf.constant([[0.0] + [-INF] * (beam_size - 1)])
    alive_log_probs = tf.tile(initial_log_probs, [batch_size, 1])
    alive_seq = _expand_to_beam_size(initial_ids, beam_size)
    alive_seq = tf.expand_dims(alive_seq, axis = 2)
    if states:
        states = nest.map_structure(
            lambda state: _expand_to_beam_size(state, beam_size), states
        )
    else:
        states = {}
