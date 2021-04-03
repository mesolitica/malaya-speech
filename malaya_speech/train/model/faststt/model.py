import tensorflow as tf
from ..conformer.layer import Conv2dSubsampling
from ..fastspeech.model import TFFastSpeechEncoder
import numpy as np


class Model(tf.keras.Model):
    def __init__(self, config, sumbsampling_filters = 144, **kwargs):
        super(Model, self).__init__(name = 'faststt', **kwargs)
        self.config = config
        self.encoder = TFFastSpeechEncoder(config, name = 'encoder')
        self.position_embeddings = tf.convert_to_tensor(
            self._sincos_embedding()
        )
        self.subsampling = Conv2dSubsampling(filters = sumbsampling_filters)
        self.encoder_dense = tf.keras.layers.Dense(
            units = config.hidden_size,
            dtype = tf.float32,
            name = 'encoder_dense',
        )

    def call(self, x, training = True):
        x = self.subsampling(x)
        x = self.encoder_dense(x)

        mel_lengths = tf.tile([tf.shape(x)[1]], [tf.shape(x)[0]])
        max_length = tf.cast(tf.reduce_max(mel_lengths), tf.int32)
        attention_mask = tf.sequence_mask(
            lengths = mel_lengths, maxlen = max_length, dtype = tf.float32
        )
        attention_mask.set_shape((None, None))

        input_shape = tf.shape(x)
        seq_length = input_shape[1]

        position_ids = tf.range(1, seq_length + 1, dtype = tf.int32)[
            tf.newaxis, :
        ]
        inputs = tf.cast(position_ids, tf.int32)
        position_embeddings = tf.gather(self.position_embeddings, inputs)
        x = x + tf.cast(position_embeddings, x.dtype)
        f = self.encoder([x, attention_mask], training = training)[0]
        return f

    def _sincos_embedding(self):
        position_enc = np.array(
            [
                [
                    pos
                    / np.power(10000, 2.0 * (i // 2) / self.config.hidden_size)
                    for i in range(self.config.hidden_size)
                ]
                for pos in range(self.config.max_position_embeddings + 1)
            ]
        )

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])

        # pad embedding.
        position_enc[0] = 0.0

        return position_enc
