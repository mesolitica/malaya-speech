import tensorflow as tf


def generate_guided_attention(
    mel_lengths, input_lengths, reduction_factor = 1, g = 0.2
):
    mel_len = mel_lengths // reduction_factor
    char_len = input_lengths
    xv, yv = tf.meshgrid(tf.range(char_len), tf.range(mel_len), indexing = 'ij')
    f32_matrix = tf.cast(yv / mel_len - xv / char_len, tf.float32)
    a = 1.0 - tf.math.exp(-(f32_matrix ** 2) / (2 * g ** 2))
    return a


from .config import Config
from .model import Model
