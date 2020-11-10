import tensorflow as tf


def get_initializer(initializer_range = 0.02):
    """Creates a `tf.initializers.truncated_normal` with the given range.
    Args:
        initializer_range: float, initializer range for stddev.
    Returns:
        TruncatedNormal initializer with stddev = `initializer_range`.
    """
    return tf.keras.initializers.TruncatedNormal(
        mean = 0.0, stddev = initializer_range
    )


def gelu(x):
    """Gaussian Error Linear unit."""
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))
    return x * cdf


def gelu_new(x):
    """Smoother gaussian Error Linear Unit."""
    cdf = 0.5 * (
        1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
    )
    return x * cdf


def swish(x):
    """Swish activation function."""
    return tf.nn.swish(x)


def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


ACT2FN = {
    'identity': tf.keras.layers.Activation('linear'),
    'tanh': tf.keras.layers.Activation('tanh'),
    'gelu': tf.keras.layers.Activation(gelu),
    'relu': tf.keras.activations.relu,
    'swish': tf.keras.layers.Activation(swish),
    'gelu_new': tf.keras.layers.Activation(gelu_new),
    'mish': tf.keras.layers.Activation(mish),
}
