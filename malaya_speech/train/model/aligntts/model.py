import tensorflow as tf
from ..utils import shape_list
from ..fastspeech2.model import FastSpeechVariantPredictor
from ..fastspeech.layer import get_initializer
from ..fastspeech.model import Model as FastSpeech


class Encoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(Encoder, self).__init__(name='Encoder', **kwargs)
        self.embeddings = TFFastSpeechEmbeddings(config, name='embeddings')
        self.encoder = TFFastSpeechEncoder(
            config.encoder_self_attention_params, name='encoder'
        )

    def call(
        self,
        input_ids,
        training=True,
        **kwargs,
    ):
        speaker_ids = tf.convert_to_tensor([0], tf.int32)
        attention_mask = tf.math.not_equal(input_ids, 0)
        embedding_output = self.embeddings(
            [input_ids, speaker_ids], training=training
        )
        attention_mask = tf.cast(attention_mask, tf.float32)
        attention_mask_expand = tf.expand_dims(attention_mask, -1)
        x = self.encoder(
            [embedding_output, attention_mask], training=training
        )
        return x
