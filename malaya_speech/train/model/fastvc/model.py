import tensorflow as tf
from ..fastspeech.model import TFFastSpeechEncoder


class Model(tf.keras.Model):
    def __init__(self, dim_neck, dim_pre, freq, **kwargs):
        super(Model, self).__init__(name = 'fastvc', **kwargs)

    def call(self, x, c_org, c_trg, training = True, **kwargs):
        attention_mask = tf.math.not_equal(x, 0)
