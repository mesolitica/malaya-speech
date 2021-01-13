import numpy as np
import tensorflow as tf


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)
