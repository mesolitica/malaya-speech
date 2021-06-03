from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.nn_ops import dropout
from tensorflow.python.ops.rnn_cell import ResidualWrapper, DropoutWrapper


class ZoneoutWrapper(rnn_cell_impl.RNNCell):
    """Operator adding zoneout to all states (states+cells) of the given cell.
  Code taken from https://github.com/teganmaharaj/zoneout
  applying zoneout as described in https://arxiv.org/pdf/1606.01305.pdf"""

    def __init__(self, cell, zoneout_prob, is_training=True, seed=None):
        if not isinstance(cell, rnn_cell_impl.RNNCell):
            raise TypeError('The parameter cell is not an RNNCell.')
        if isinstance(zoneout_prob, float) and not (
            zoneout_prob >= 0.0 and zoneout_prob <= 1.0
        ):
            raise ValueError(
                'Parameter zoneout_prob must be between 0 and 1: %d'
                % zoneout_prob
            )
        self._cell = cell
        self._zoneout_prob = (zoneout_prob, zoneout_prob)
        self._seed = seed
        self._is_training = is_training

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        if isinstance(self.state_size, tuple) != isinstance(
            self._zoneout_prob, tuple
        ):
            raise TypeError('Subdivided states need subdivided zoneouts.')
        if isinstance(self.state_size, tuple) and len(
            tuple(self.state_size)
        ) != len(tuple(self._zoneout_prob)):
            raise ValueError('State and zoneout need equally many parts.')
        output, new_state = self._cell(inputs, state, scope)
        if isinstance(self.state_size, tuple):
            if self._is_training:
                new_state = tuple(
                    (1 - state_part_zoneout_prob)
                    * dropout(
                        new_state_part - state_part,
                        (1 - state_part_zoneout_prob),
                        seed=self._seed,
                    )
                    + state_part
                    for new_state_part, state_part, state_part_zoneout_prob in zip(
                        new_state, state, self._zoneout_prob
                    )
                )
            else:
                new_state = tuple(
                    state_part_zoneout_prob * state_part
                    + (1 - state_part_zoneout_prob) * new_state_part
                    for new_state_part, state_part, state_part_zoneout_prob in zip(
                        new_state, state, self._zoneout_prob
                    )
                )
            new_state = rnn_cell_impl.LSTMStateTuple(new_state[0], new_state[1])
        else:
            raise ValueError('Only states that are tuples are supported')
        return output, new_state


def single_cell(
    cell_class,
    cell_params,
    dp_input_keep_prob=1.0,
    dp_output_keep_prob=1.0,
    recurrent_keep_prob=1.0,
    input_weight_keep_prob=1.0,
    recurrent_weight_keep_prob=1.0,
    weight_variational=False,
    dropout_seed=None,
    zoneout_prob=0.0,
    training=True,
    residual_connections=False,
    awd_initializer=False,
    variational_recurrent=False,  # in case they want to use DropoutWrapper
    dtype=None,
):
    """Creates an instance of the rnn cell.
     Such cell describes one step one layer and can include residual connection
     and/or dropout
     Args:
      cell_class: Tensorflow RNN cell class
      cell_params (dict): cell parameters
      dp_input_keep_prob (float): (default: 1.0) input dropout keep
        probability.
      dp_output_keep_prob (float): (default: 1.0) output dropout keep
        probability.
      zoneout_prob(float): zoneout probability. Applying both zoneout and
        droupout is currently not supported
      residual_connections (bool): whether to add residual connection
     Returns:
       TF RNN instance
  """
    if awd_initializer:
        val = 1.0 / math.sqrt(cell_params['num_units'])
        cell_params['initializer'] = tf.random_uniform_initializer(
            minval=-val, maxval=val
        )

    cell = cell_class(**cell_params)
    if residual_connections:
        cell = ResidualWrapper(cell)
    if zoneout_prob > 0.0 and (
        dp_input_keep_prob < 1.0 or dp_output_keep_prob < 1.0
    ):
        raise ValueError(
            'Currently applying both dropout and zoneout on the same cell.'
            'This is currently not supported.'
        )
    if dp_input_keep_prob != 1.0 or dp_output_keep_prob != 1.0 and training:
        cell = DropoutWrapper(
            cell,
            input_keep_prob=dp_input_keep_prob,
            output_keep_prob=dp_output_keep_prob,
            variational_recurrent=variational_recurrent,
            dtype=dtype,
            seed=dropout_seed,
        )
    if zoneout_prob > 0.0:
        cell = ZoneoutWrapper(cell, zoneout_prob, is_training=training)
    return cell
