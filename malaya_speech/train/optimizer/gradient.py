import tensorflow as tf


def scale_gradient(tensor: tf.Tensor, scale: float) -> tf.Tensor:
    """
    Scale gradients for reverse differentiation proportional to the given scale.
    Does not influence the magnitude/ scale of the output from a given tensor (just the gradient).
    eg, loss = scale_gradient(loss, 0.5)
    """
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)
