import tensorflow as tf


class Flip(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, x, mask):
        x = tf.reverse(x, [2])
        logdet = tf.zeros([tf.shape(x)[0]])
        return x, logdet

    def inverse(self, x, mask):
        x = tf.reverse(x, [2])
        return x
