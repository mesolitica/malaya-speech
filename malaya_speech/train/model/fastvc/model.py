import tensorflow as tf
from ..fastspeech.model import TFFastSpeechEncoder, TFTacotronPostnet


class Encoder(tf.keras.layers.Layer):
    def __init__(self, config, freq, **kwargs):
        super(Encoder, self).__init__(name = 'Encoder', **kwargs)
        self.freq = freq
        self.encoder = TFFastSpeechEncoder(config, name = 'encoder')
        self.dense = tf.keras.layers.Dense(config.hidden_size)

    def call(self, x, c_org, attention_mask, training = True):
        c_org = tf.tile(tf.expand_dims(c_org, 1), (1, tf.shape(x)[1], 1))
        x = tf.concat([x, c_org], axis = -1)
        x = self.dense(x)
        x_reversed = tf.reverse(x, [1])
        attention_mask_reversed = tf.reverse(attention_mask, [1])
        out_forward = self.encoder([x, attention_mask])[0]
        out_backward = self.encoder([x_reversed, attention_mask_reversed])[0]

        lens = tf.shape(out_forward)[1]

        codes = tf.TensorArray(
            dtype = tf.float32,
            size = 0,
            dynamic_size = True,
            infer_shape = True,
        )
        init_state = (0, 0, codes)

        def condition(i, counter, codes):
            return i < lens

        def body(i, counter, codes):
            c = tf.concat(
                [out_forward[:, i + self.freq - 1, :], out_backward[:, i, :]],
                axis = -1,
            )
            return i + self.freq, counter + 1, codes.write(counter, c)

        _, _, codes = tf.while_loop(condition, body, init_state)
        codes = codes.stack()

        return codes


class Decoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(Decoder, self).__init__(name = 'Decoder', **kwargs)
        self.encoder = TFFastSpeechEncoder(config, name = 'encoder')
        self.dense = tf.keras.layers.Dense(config.hidden_size)

    def call(self, x, attention_mask, training = True):
        x = self.dense(x)
        return self.encoder([x, attention_mask])[0]


class Model(tf.keras.Model):
    def __init__(self, config, freq, **kwargs):
        super(Model, self).__init__(name = 'fastvc', **kwargs)
        self.encoder = Encoder(config.encoder_self_attention_params, freq)
        self.decoder = Decoder(config.decoder_self_attention_params)
        self.mel_dense = tf.keras.layers.Dense(
            units = config.num_mels, dtype = tf.float32, name = 'mel_before'
        )
        self.postnet = TFTacotronPostnet(
            config = config, dtype = tf.float32, name = 'postnet'
        )
        self.config = config

    def call_second(self, x, c_org, mel_lengths, training = True):
        max_length = tf.cast(tf.reduce_max(mel_lengths), tf.int32)
        attention_mask = tf.sequence_mask(
            lengths = mel_lengths, maxlen = max_length, dtype = tf.float32
        )
        attention_mask.set_shape((None, None))
        return self.encoder(x, c_org, attention_mask, training = training)

    def call(self, x, c_org, c_trg, mel_lengths, training = True, **kwargs):

        max_length = tf.cast(tf.reduce_max(mel_lengths), tf.int32)
        attention_mask = tf.sequence_mask(
            lengths = mel_lengths, maxlen = max_length, dtype = tf.float32
        )
        attention_mask.set_shape((None, None))
        codes = self.encoder(
            x, c_org, attention_mask, training = training
        )  # [STACK, B, D]

        tmp = tf.TensorArray(
            dtype = tf.float32,
            size = 0,
            dynamic_size = True,
            infer_shape = True,
        )
        init_state = (0, tmp)
        stack_size = tf.shape(codes)[0]

        def condition(i, tmp):
            return i < stack_size

        def body(i, tmp):
            c = tf.expand_dims(codes[i], 1)
            c = tf.tile(
                c, (1, tf.cast(tf.shape(x)[1] / stack_size, tf.int32), 1)
            )
            return i + 1, tmp.write(i, c)

        _, tmp = tf.while_loop(condition, body, init_state)
        tmp = tmp.stack()  # [STACK, B, T, D]
        code_exp = tf.reshape(
            tmp,
            (
                tf.shape(x)[0],
                -1,
                self.config.encoder_self_attention_params.hidden_size * 2,
            ),
        )
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

        return encoder_outputs, mel_before, mel_after, codes
