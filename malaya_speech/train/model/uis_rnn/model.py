import tensorflow as tf

_INITIAL_SIGMA2_VALUE = 0.1


class CoreRNN(tf.keras.layers.Layer):
    def __init__(
        self,
        observation_dim = 256,
        rnn_hidden_size = 512,
        rnn_depth = 1,
        rnn_dropout = 0.0,
        rnn_cell = tf.keras.layers.GRU,
        **kwargs,
    ):
        super(CoreRNN, self).__init__(name = 'CoreRNN', **kwargs)
        self.lstm = tf.keras.Sequential()
        for i in range(rnn_depth):
            self.lstm.add(
                tf.keras.layers.LSTM(
                    rnn_hidden_size,
                    return_sequences = True,
                    return_state = True,
                )
            )
        self.linear_mean1 = tf.keras.layers.Dense(
            units = hidden_size, dtype = tf.float32, activation = tf.nn.relu
        )
        self.linear_mean2 = tf.keras.layers.Dense(
            units = observation_dim, dtype = tf.float32
        )

    def call(self, x, hidden = None, training = True):
        output_seq, hidden = self.lstm(
            x, training = training, initial_state = hidden
        )
        mean = self.linear_mean2(self.linear_mean1(output_seq))

        return mean, hidden


class BeamState:
    """Structure that contains necessary states for beam search."""

    def __init__(self, source = None):
        if not source:
            self.mean_set = []
            self.hidden_set = []
            self.neg_likelihood = 0
            self.trace = []
            self.block_counts = []
        else:
            self.mean_set = source.mean_set.copy()
            self.hidden_set = source.hidden_set.copy()
            self.trace = source.trace.copy()
            self.block_counts = source.block_counts.copy()
            self.neg_likelihood = source.neg_likelihood

    def append(self, mean, hidden, cluster):
        """Append new item to the BeamState."""
        self.mean_set.append(mean.clone())
        self.hidden_set.append(hidden.clone())
        self.block_counts.append(1)
        self.trace.append(cluster)


class Model(tf.keras.Model):
    def __init__(
        self,
        observation_dim = 256,
        rnn_hidden_size = 512,
        rnn_depth = 1,
        rnn_dropout = 0.0,
        sigma2 = None,
        transition_bias = None,
        crp_alpha = 1.0,
        **kwargs,
    ):
        super(Model, self).__init__(name = 'uis-rnn', **kwargs)
        self.rnn_model = CoreRNN(
            observation_dim, rnn_hidden_size, rnn_depth, rnn_dropout
        )
        self.estimate_sigma2 = sigma2 is None
        self.estimate_transition_bias = transition_bias is None
        sigma2 = _INITIAL_SIGMA2_VALUE if self.estimate_sigma2 else args.sigma2
        self.sigma2 = sigma2 * tf.get_variable(
            name = 'sigma2',
            shape = [observation_dim],
            initializer = tf.ones_initializer(),
        )
        self.transition_bias = transition_bias
        self.transition_bias_denominator = 0.0
        self.crp_alpha = crp_alpha

    def call(
        self,
        train_sequences,
        train_cluster_ids,
        transition_bias,
        transition_bias_denominator,
        concatenated_train_sequence,
        concatenated_train_cluster_id,
    ):

        if self.estimate_transition_bias:
            if self.transition_bias is None:
                self.transition_bias = transition_bias
                self.transition_bias_denominator = transition_bias_denominator
            else:
                self.transition_bias = (
                    self.transition_bias * self.transition_bias_denominator
                    + transition_bias * transition_bias_denominator
                ) / (
                    self.transition_bias_denominator
                    + transition_bias_denominator
                )
                self.transition_bias_denominator += transition_bias_denominator