import tensorflow as tf
from ..fastspeech.model import TFFastSpeechEncoder
import numpy as np


class Encoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(Encoder, self).__init__(name = 'Encoder', **kwargs)
        self.config = config
        self.encoder = TFFastSpeechEncoder(config, name = 'encoder')
        self.position_embeddings = tf.convert_to_tensor(
            self._sincos_embedding()
        )

    def call(self, x, attention_mask, training = True):
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


class Model(tf.keras.Model):
    def __init__(
        self, config, O, C, kernel_size = 5, masking = False, **kwargs
    ):
        super(Model, self).__init__(name = 'fastvc', **kwargs)
        self.encoder = Encoder(config.encoder_self_attention_params)
        self.decoder = Encoder(config.decoder_self_attention_params)
        self.encoder_dense = tf.keras.layers.Conv1D(
            config.encoder_self_attention_params.hidden_size,
            kernel_size = kernel_size,
            strides = 1,
            use_bias = False,
            padding = 'SAME',
        )
        self.mel_dense = tf.keras.layers.Dense(
            units = config.num_mels, dtype = tf.float32, name = 'mel_before'
        )
        self.dim = O
        self.C = C
        self.masking = masking

    def call(self, x, mel_lengths, training = True, **kwargs):
        original = x
        T_mix = tf.shape(x)[1]
        batch_size = tf.shape(x)[0]
        max_length = tf.cast(tf.reduce_max(mel_lengths), tf.int32)
        attention_mask = tf.sequence_mask(
            lengths = mel_lengths, maxlen = max_length, dtype = tf.float32
        )
        attention_mask.set_shape((None, None))
        x = tf.concat([x] * self.C, axis = 2)
        x = self.encoder_dense(x)
        encoded = self.encoder(x, attention_mask, training = training)
        decoder_output = self.decoder(
            encoded, attention_mask, training = training
        )
        decoder_output = tf.reshape(
            decoder_output, (batch_size, T_mix, self.C, self.dim)
        )
        mel_before = self.mel_dense(decoder_output)
        if self.masking:
            mel_before = tf.nn.tanh(mel_before)
            tiled = tf.tile(tf.expand_dims(original, 2), [1, 1, self.C, 1])
            return tiled * mel_before
        else:
            return mel_before
