import tensorflow as tf
from ..fastvc.model import Decoder
from ..fastspeech.model import TFTacotronPostnet


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
    def __init__(self, dim_neck, freq, **kwargs):
        super(Encoder, self).__init__(name = 'Encoder', **kwargs)
        self.dim_neck = dim_neck
        self.freq = freq

        self.convolutions = []
        for i in range(3):
            convolutions = tf.keras.Sequential()
            convolutions.add(
                ConvNorm(512, kernel_size = 5, stride = 1, dilation = 1)
            )
            convolutions.add(tf.keras.layers.BatchNormalization())
            self.convolutions.append(convolutions)

        self.lstm = tf.keras.Sequential()
        for i in range(2):
            self.lstm.add(
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(dim_neck, return_sequences = True)
                )
            )

    def call(self, x, c_org, training = True):
        c_org = tf.tile(tf.expand_dims(c_org, 1), (1, tf.shape(x)[1], 1))
        x = tf.concat([x, c_org], axis = -1)
        for c in self.convolutions:
            x = c(x, training = training)
            x = tf.nn.tanh(x)
        outputs = self.lstm(x)
        out_forward = outputs[:, :, : self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck :]

        lens = tf.shape(outputs)[1]

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


class Model(tf.keras.Model):
    def __init__(self, dim_neck, dim_pre, freq, config, **kwargs):
        super(Model, self).__init__(name = 'autovc', **kwargs)
        self.encoder = Encoder(dim_neck, freq)
        self.decoder = Decoder(config.decoder_self_attention_params)
        self.mel_dense = tf.keras.layers.Dense(
            units = config.num_mels, dtype = tf.float32, name = 'mel_before'
        )
        self.postnet = TFTacotronPostnet(
            config = config, dtype = tf.float32, name = 'postnet'
        )

    def call_second(self, x, c_org, training = True):
        return self.encoder(x, c_org, training = training)

    def call(self, x, c_org, c_trg, mel_lengths, training = True, **kwargs):

        max_length = tf.cast(tf.reduce_max(mel_lengths), tf.int32)
        attention_mask = tf.sequence_mask(
            lengths = mel_lengths, maxlen = max_length, dtype = tf.float32
        )
        attention_mask.set_shape((None, None))

        codes = self.encoder(x, c_org, training = training)  # [STACK, B, D]

        tmp = tf.zeros(
            shape = [tf.shape(x)[0], 0, self.encoder.dim_neck * 2],
            dtype = tf.float32,
        )
        i = tf.constant(0, dtype = tf.int32)
        stack_size = tf.shape(codes)[0]

        def condition(i, tmp):
            return i < stack_size

        def body(i, tmp):
            c = tf.expand_dims(codes[i], 1)
            c = tf.tile(
                c, (1, tf.cast(tf.shape(x)[1] / stack_size, tf.int32), 1)
            )
            return i + 1, tf.concat([tmp, c], axis = 1)

        _, code_exp = tf.while_loop(
            condition,
            body,
            [i, tmp],
            shape_invariants = [
                i.get_shape(),
                tf.TensorShape([None, None, self.encoder.dim_neck * 2]),
            ],
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
