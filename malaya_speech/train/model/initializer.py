from tensorflow.compat.v1.keras.initializers import VarianceScaling


class HeNormal(VarianceScaling):
    """He normal initializer.
     Also available via the shortcut function
    `tf.keras.initializers.he_normal`.
    It draws samples from a truncated normal distribution centered on 0 with
    `stddev = sqrt(2 / fan_in)` where `fan_in` is the number of input units in the
    weight tensor.
    Examples:
    >>> # Standalone usage:
    >>> initializer = tf.keras.initializers.HeNormal()
    >>> values = initializer(shape=(2, 2))
    >>> # Usage in a Keras layer:
    >>> initializer = tf.keras.initializers.HeNormal()
    >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)
    Args:
      seed: A Python integer. Used to make the behavior of the initializer
        deterministic. Note that a seeded
        initializer will not produce the same random values across multiple calls,
        but multiple initializers will produce the same sequence when constructed
        with the same seed value.
    References:
      - [He et al., 2015](https://arxiv.org/abs/1502.01852)
    """

    def __init__(self, seed=None):
        super(HeNormal, self).__init__(
            scale=2., mode='fan_in', distribution='truncated_normal', seed=seed)

    def get_config(self):
        return {'seed': self.seed}
