import tensorflow as tf
from ..fastspeech.model import TFFastSpeechEncoder, TFTacotronPostnet
import numpy as np


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        dim_neck,
        dim_speaker,
        dim_input,
        skip,
        config,
        use_position_embedding=True,
        **kwargs
    ):
        super(Encoder, self).__init__(name='Encoder', **kwargs)
        self.config = config
        self.encoder = TFFastSpeechEncoder(config, name='encoder')

        self.encoder_dense = tf.keras.layers.Dense(
            units=dim_neck, dtype=tf.float32, name='encoder_dense'
        )
        self.use_position_embedding = use_position_embedding
        if self.use_position_embedding:
            self.position_embeddings = tf.convert_to_tensor(
                self._sincos_embedding()
            )

        a = []
        index = 0
        for i in range(dim_speaker):
            n = [i + dim_input]
            if index < dim_input and i % skip == 0:
                n.append(index)
                index += 1
            a.extend(n)
        self.indices = a

    def call(self, x, c_org, attention_mask, training=True):
        c_org = tf.tile(tf.expand_dims(c_org, 1), (1, tf.shape(x)[1], 1))
        x = tf.concat([x, c_org], axis=-1)
        x = tf.gather(x, self.indices, axis=-1)
        input_shape = tf.shape(x)
        seq_length = input_shape[1]

        if self.use_position_embedding:
            position_ids = tf.range(1, seq_length + 1, dtype=tf.int32)[
                tf.newaxis, :
            ]
            inputs = tf.cast(position_ids, tf.int32)
            position_embeddings = tf.gather(self.position_embeddings, inputs)
            x = x + tf.cast(position_embeddings, x.dtype)

        f = self.encoder([x, attention_mask], training=training)[0]
        return self.encoder_dense(f)

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

        position_enc[0] = 0.0

        return position_enc


class Decoder(tf.keras.layers.Layer):
    def __init__(self, config, use_position_embedding=True, **kwargs):
        super(Decoder, self).__init__(name='Decoder', **kwargs)
        self.config = config
        self.encoder = TFFastSpeechEncoder(config, name='encoder')
        self.use_position_embedding = use_position_embedding
        if self.use_position_embedding:
            self.position_embeddings = tf.convert_to_tensor(
                self._sincos_embedding()
            )

    def call(self, x, attention_mask, training=True):
        input_shape = tf.shape(x)
        seq_length = input_shape[1]

        if self.use_position_embedding:
            position_ids = tf.range(1, seq_length + 1, dtype=tf.int32)[
                tf.newaxis, :
            ]
            inputs = tf.cast(position_ids, tf.int32)
            position_embeddings = tf.gather(self.position_embeddings, inputs)
            x = x + tf.cast(position_embeddings, x.dtype)

        return self.encoder([x, attention_mask], training=training)[0]

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
        self,
        dim_neck,
        config,
        dim_input=80,
        dim_speaker=512,
        skip=4,
        use_position_embedding=False,
        **kwargs
    ):
        super(Model, self).__init__(name='fastvc2', **kwargs)
        self.encoder = Encoder(
            dim_neck=dim_neck,
            dim_speaker=dim_speaker,
            dim_input=dim_input,
            skip=skip,
            config=config.encoder_self_attention_params,
            use_position_embedding=use_position_embedding,
        )
        self.decoder = Decoder(
            config.decoder_self_attention_params,
            use_position_embedding=use_position_embedding,
        )
        self.mel_dense = tf.keras.layers.Dense(
            units=config.num_mels, dtype=tf.float32, name='mel_before'
        )
        self.postnet = TFTacotronPostnet(
            config=config, dtype=tf.float32, name='postnet'
        )
        self.config = config

        a = []
        index = 0
        for i in range(dim_speaker):
            n = [i + dim_neck]
            if index < dim_neck and i % skip == 0:
                n.append(index)
                index += 1
            a.extend(n)
        self.indices = a

    def call_second(self, x, c_org, mel_lengths, training=True):
        max_length = tf.cast(tf.reduce_max(mel_lengths), tf.int32)
        attention_mask = tf.sequence_mask(
            lengths=mel_lengths, maxlen=max_length, dtype=tf.float32
        )
        attention_mask.set_shape((None, None))
        return self.encoder(x, c_org, attention_mask, training=training)

    def call(self, x, c_org, c_trg, mel_lengths, training=True, **kwargs):

        max_length = tf.cast(tf.reduce_max(mel_lengths), tf.int32)
        attention_mask = tf.sequence_mask(
            lengths=mel_lengths, maxlen=max_length, dtype=tf.float32
        )
        attention_mask.set_shape((None, None))
        code_exp = self.encoder(x, c_org, attention_mask, training=training)
        c_trg = tf.tile(tf.expand_dims(c_trg, 1), (1, tf.shape(x)[1], 1))
        encoder_outputs = tf.concat([code_exp, c_trg], axis=-1)
        encoder_outputs = tf.gather(encoder_outputs, self.indices, axis=-1)
        decoder_output = self.decoder(
            encoder_outputs, attention_mask, training=training
        )
        mel_before = self.mel_dense(decoder_output)
        mel_after = (
            self.postnet([mel_before, attention_mask], training=training)
            + mel_before
        )

        return encoder_outputs, mel_before, mel_after, code_exp
