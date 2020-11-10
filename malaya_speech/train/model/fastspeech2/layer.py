import tensorflow as tf


def get_initializer(initializer_seed = 42):
    """
    Creates a `tf.initializers.he_normal` with the given seed.
    Args:
        initializer_seed: int, initializer seed.
    Returns:
        HeNormal initializer with seed = `initializer_seed`.
    """
    return tf.keras.initializers.he_normal(seed = initializer_seed)
