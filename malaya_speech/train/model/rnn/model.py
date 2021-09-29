import tensorflow as tf
from ..utils import shape_list
from ..hubert.layer import ConvFeatureExtractionModel


class TimeReduction(tf.keras.layers.Layer):
    def __init__(self, reduction_factor=2, **kwargs):
        self.reduction_factor = reduction_factor

    def call(self, xs):
        batch_size, xlen, hidden_size = shape_list(xs)
        reduction_factor = xlen % self.reduction_factor
        reduction_factor = tf.cond(tf.math.not_equal(reduction_factor, 0),
                                   lambda: self.reduction_factor - reduction_factor,
                                   lambda: reduction_factor)
        xs = tf.pad(xs, [(0, 0), (reduction_factor, 0), (0, 0)])
        xs = tf.reshape(x, (batch_size, -1, self.reduction_factor, hidden_size))
        xs = tf.reduce_mean(xs, axis=2)
        return xs


class ResLayerNormLSTM(tf.keras.layers.Layer):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 dropout=0,
                 time_reductions=[1],
                 reduction_factor=2, **kwargs):
        super(ResLayerNormLSTM, self).__init__(name='ResLayerNormLSTM', **kwargs)
        self.hidden_size = hidden_size
        self.lstms = []
        self.projs = []
        for i in range(num_layers):
            rnn = tf.keras.layers.LSTM(
                units=self.hidden_size,
                return_sequences=True,
                name=f'encoder_lstm_{i}',
                return_state=True,
            )
            self.lstms.append(rnn)
            proj = [tf.keras.layers.LayerNormalization(name=f'encoder_ln_{i}')]
            if i in time_reductions:
                proj.append(TimeReduction(reduction_factor))
            if dropout > 0:
                proj.append(tf.keras.layers.Dropout(dropout))
            self.projs.append(tf.keras.Sequential(proj))

    def get_initial_state(self):
        """Get zeros states
        Returns:
            tf.Tensor: states having shape [num_rnns, 1 or 2, B, P]
        """
        states = []
        for rnn in self.rnns:
            states.append(
                tf.stack(
                    rnn.get_initial_state(
                        tf.zeros([1, 1, 1], dtype=tf.float32)
                    ),
                    axis=0,
                )
            )
        return tf.stack(states, axis=0)

    def call(self, xs, training=True, hiddens=None):
        if hiddens is None:
            o = get_initial_state()

        new_states = []
        for i, (lstm, proj) in enumerate(zip(self.lstms, self.projs)):
            outputs = lstm(
                outputs,
                training=training,
                initial_state=tf.unstack(states[i], axis=0),
            )
            new_states.append(tf.stack(outputs[1:]))
            outputs = outputs[0]
            xs_next, (h, c) = lstm(xs, (hs[i, None], cs[i, None]))
