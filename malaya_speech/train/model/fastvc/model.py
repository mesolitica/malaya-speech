import tensorflow as tf
from ..fastspeech.model import TFFastSpeechEncoder, TFTacotronPostnet
import numpy as np


class ConvNorm(tf.keras.layers.Layer):
    def __init__(
        self,
        out_channels,
        kernel_size = 1,
        stride = 1,
        padding = 'SAME',
        dilation = 1,
        bias = True,
        **kwargs,
    ):
        super(ConvNorm, self).__init__(name = 'ConvNorm', **kwargs)
        self.conv = tf.keras.layers.Conv1D(
            out_channels,
            kernel_size = kernel_size,
            strides = stride,
            padding = padding,
            dilation_rate = dilation,
            use_bias = bias,
        )

    def call(self, x):
        return self.conv(x)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, dim_neck, config, **kwargs):
        super(Encoder, self).__init__(name = 'Encoder', **kwargs)
        self.config = config
        self.encoder = TFFastSpeechEncoder(config, name = 'encoder')
        self.position_embeddings = tf.convert_to_tensor(
            self._sincos_embedding()
        )
        self.encoder_dense = tf.keras.layers.Dense(
            units = dim_neck, dtype = tf.float32, name = 'encoder_dense'
        )

    def call(self, x, c_org, attention_mask, training = True):
        c_org = tf.tile(tf.expand_dims(c_org, 1), (1, tf.shape(x)[1], 1))
        x = tf.concat([x, c_org], axis = -1)
        input_shape = tf.shape(x)
        seq_length = input_shape[1]

        position_ids = tf.range(1, seq_length + 1, dtype = tf.int32)[
            tf.newaxis, :
        ]
        inputs = tf.cast(position_ids, tf.int32)
        position_embeddings = tf.gather(self.position_embeddings, inputs)
        x = x + tf.cast(position_embeddings, x.dtype)
        f = self.encoder([x, attention_mask], training = training)[0]
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

        # pad embedding.
        position_enc[0] = 0.0

        return position_enc


class Decoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(Decoder, self).__init__(name = 'Decoder', **kwargs)
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
        return self.encoder([x, attention_mask], training = training)[0]

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
    def __init__(self, dim_neck, config, dim_speaker = 0, **kwargs):
        super(Model, self).__init__(name = 'fastvc', **kwargs)
        self.encoder = Encoder(dim_neck, config.encoder_self_attention_params)
        self.decoder = Decoder(config.decoder_self_attention_params)
        self.mel_dense = tf.keras.layers.Dense(
            units = config.num_mels, dtype = tf.float32, name = 'mel_before'
        )
        self.postnet = TFTacotronPostnet(
            config = config, dtype = tf.float32, name = 'postnet'
        )
        self.config = config
        if dim_speaker > 0:
            self.dim_speaker = tf.keras.layers.Dense(
                units = dim_speaker, dtype = tf.float32, name = 'dim_speaker'
            )
        else:
            self.dim_speaker = None

    def call_second(self, x, c_org, mel_lengths, training = True):
        max_length = tf.cast(tf.reduce_max(mel_lengths), tf.int32)
        attention_mask = tf.sequence_mask(
            lengths = mel_lengths, maxlen = max_length, dtype = tf.float32
        )
        attention_mask.set_shape((None, None))
        if self.dim_speaker:
            c_org = self.dim_speaker(c_org)
        return self.encoder(x, c_org, attention_mask, training = training)

    def call(self, x, c_org, c_trg, mel_lengths, training = True, **kwargs):

        if self.dim_speaker:
            c_org = self.dim_speaker(c_org)
            c_trg = self.dim_speaker(c_trg)

        max_length = tf.cast(tf.reduce_max(mel_lengths), tf.int32)
        attention_mask = tf.sequence_mask(
            lengths = mel_lengths, maxlen = max_length, dtype = tf.float32
        )
        attention_mask.set_shape((None, None))
        code_exp = self.encoder(x, c_org, attention_mask, training = training)
        c_trg = tf.tile(tf.expand_dims(c_trg, 1), (1, tf.shape(x)[1], 1))
        encoder_outputs = tf.concat([code_exp, c_trg], axis = -1)
        decoder_output = self.decoder(
            encoder_outputs, attention_mask, training = training
        )
        mel_before = self.mel_dense(decoder_output)
        mel_after = (
            self.postnet([mel_before, attention_mask], training = training)
            + mel_before
        )

        return encoder_outputs, mel_before, mel_after, code_exp
