import tensorflow as tf
import numpy as np
from ..autovc.model import Postnet
from ..utils import GroupNormalization


def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


def quantize_f0_numpy(x, num_bins=256):
    # x is logf0
    assert x.ndim == 1
    x = x.astype(float).copy()
    uv = x <= 0
    x[uv] = 0.0
    x = np.round((x / np.max(x)) * (num_bins - 1))
    x = x + 1
    x[uv] = 0.0
    enc = np.zeros((len(x), num_bins + 1), dtype=np.float32)
    enc[np.arange(len(x)), x.astype(np.int32)] = 1.0
    return enc, x.astype(np.int64)


def quantize_f0_tf(x, num_bins=256):
    batch_size = tf.shape(x)[0]
    x = tf.reshape(x, (-1,))
    uv = x <= 0
    x = tf.where(uv, tf.zeros_like(x), x)
    x = tf.cast(tf.round((x / tf.reduce_max(x)) * (num_bins - 1)), tf.int32)
    x = x + 1
    x = tf.where(uv, tf.zeros_like(x), x)
    enc = tf.one_hot(x, num_bins + 1)
    return tf.reshape(enc, (batch_size, -1, num_bins + 1))


class LinearNorm(tf.keras.layers.Layer):
    def __init__(self, out_dim, bias=True, **kwargs):
        super(LinearNorm, self).__init__(name='LinearNorm', **kwargs)
        self.linear_layer = tf.keras.layers.Dense(out_dim, use_bias=bias)

    def call(self, x):
        return self.linear_layer(x)


class ConvNorm(tf.keras.layers.Layer):
    def __init__(
        self,
        out_channels,
        kernel_size=1,
        stride=1,
        padding='SAME',
        dilation=1,
        bias=True,
        **kwargs,
    ):
        super(ConvNorm, self).__init__(name='ConvNorm', **kwargs)
        self.conv = tf.keras.layers.Conv1D(
            out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            dilation_rate=dilation,
            use_bias=bias,
        )

    def call(self, x):
        return self.conv(x)


class GroupNorm(tf.keras.layers.Layer):
    def __init__(self, group, **kwargs):
        super(GroupNorm, self).__init__(name='GroupNorm', **kwargs)
        self.group = group

    def call(self, input):
        # input = NWC, [B, T, D],
        return tf.contrib.layers.group_norm(
            input,
            groups=self.group,
            epsilon=1e-8,
            channels_axis=-1,
            reduction_axes=(-2,),
        )


class InterpLnr(tf.keras.layers.Layer):
    def __init__(self, hparams, **kwargs):
        super().__init__(**kwargs)
        self.max_len_seq = hparams.max_len_seq
        self.max_len_pad = hparams.max_len_pad

        self.min_len_seg = hparams.min_len_seg
        self.max_len_seg = hparams.max_len_seg

        self.max_num_seg = self.max_len_seq // self.min_len_seg + 1

    def call(self, x, len_seq, training=True):

        if not training:
            return x

        max_len_seq = tf.reduce_max(len_seq)
        max_num_seg = (max_len_seq) // self.min_len_seg + 1

        batch_size = tf.shape(x)[0]
        dim = x.shape[2]
        indices = tf.tile(
            tf.expand_dims(tf.range(self.max_len_seg * 2), 0),
            (batch_size * max_num_seg, 1),
        )
        scales = tf.random.uniform(shape=(batch_size * max_num_seg,)) + 0.5
        idx_scaled = tf.cast(indices, tf.float32) / tf.expand_dims(scales, -1)
        idx_scaled_fl = tf.math.floor(idx_scaled)
        lambda_ = idx_scaled - idx_scaled_fl
        len_seg = tf.random.uniform(
            (batch_size * max_num_seg, 1),
            minval=self.min_len_seg,
            maxval=self.max_len_seg,
            dtype=tf.int32,
        )
        idx_mask = idx_scaled_fl < (tf.cast(len_seg, tf.float32) - 1)
        offset = tf.math.cumsum(
            tf.reshape(len_seg, (batch_size, -1)), axis=-1
        )
        offset = tf.reshape(tf.pad(offset[:, :-1], ((0, 0), (1, 0))), (-1, 1))
        idx_scaled_org = idx_scaled_fl + tf.cast(offset, tf.float32)

        len_seq_rp = tf.repeat(len_seq, max_num_seg)
        idx_mask_org = idx_scaled_org < tf.cast(
            tf.expand_dims(len_seq_rp - 1, -1), tf.float32
        )

        idx_mask_final = tf.cast(idx_mask & idx_mask_org, tf.int32)

        counts = tf.reduce_sum(
            tf.reshape(
                tf.reduce_sum(idx_mask_final, axis=-1), (batch_size, -1)
            ),
            axis=-1,
        )

        index_1 = tf.repeat(tf.range(batch_size), counts)

        index_2_fl = tf.cast(
            tf.boolean_mask(idx_scaled_org, idx_mask_final), tf.int32
        )
        index_2_cl = index_2_fl + 1
        concatenated = tf.transpose(
            tf.concat(
                [
                    tf.expand_dims(index_1, axis=0),
                    tf.expand_dims(index_2_fl, axis=0),
                ],
                axis=0,
            )
        )
        y_fl = tf.cast(tf.gather_nd(x, concatenated), tf.float32)

        concatenated = tf.transpose(
            tf.concat(
                [
                    tf.expand_dims(index_1, axis=0),
                    tf.expand_dims(index_2_cl, axis=0),
                ],
                axis=0,
            )
        )
        y_cl = tf.cast(tf.gather_nd(x, concatenated), tf.float32)
        lambda_f = tf.expand_dims(tf.boolean_mask(lambda_, idx_mask_final), -1)

        y = (1 - lambda_f) * y_fl + lambda_f * y_cl
        sequences = tf.reshape(
            y[: (tf.shape(y)[0] // batch_size) * batch_size],
            (batch_size, -1, dim),
        )

        sequences = sequences[:, :max_len_seq]
        return tf.pad(
            sequences,
            ((0, 0), (0, max_len_seq - tf.shape(sequences)[1]), (0, 0)),
        )


class Encoder_t(tf.keras.layers.Layer):
    def __init__(self, hparams, **kwargs):
        super(Encoder_t, self).__init__(name='Encoder_t', **kwargs)
        self.dim_neck_2 = hparams.dim_neck_2
        self.freq_2 = hparams.freq_2
        self.dim_freq = hparams.dim_freq
        self.dim_enc_2 = hparams.dim_enc_2
        self.dim_emb = hparams.dim_spk_emb
        self.chs_grp = hparams.chs_grp

        self.convolutions = []
        for i in range(1):
            convolutions = tf.keras.Sequential()
            convolutions.add(
                ConvNorm(self.dim_enc_2, kernel_size=5, stride=1)
            )
            convolutions.add(
                GroupNormalization(groups=self.dim_enc_2 // self.chs_grp)
            )
            self.convolutions.append(convolutions)

        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.dim_neck_2, return_sequences=True)
        )

    def call(self, x, mask, training=True):

        for conv in self.convolutions:
            x = mish(conv(x))

        outputs = self.lstm(x)
        print(outputs.shape)
        if mask is not None:
            outputs = outputs * mask

        out_forward = outputs[:, :, : self.dim_neck_2]
        out_backward = outputs[:, :, self.dim_neck_2:]

        out_forward = out_forward[:, self.freq_2 - 1:: self.freq_2, :]
        out_backward = out_backward[:, :: self.freq_2, :]

        len_forward = tf.shape(out_forward)[1]
        len_backward = tf.shape(out_backward)[1]
        out_backward = tf.cond(
            len_backward < len_forward,
            lambda: tf.pad(
                out_backward, ((0, 0), (0, len_forward - len_backward), (0, 0))
            ),
            lambda: out_backward,
        )
        out_forward = tf.cond(
            len_forward < len_backward,
            lambda: tf.pad(
                out_forward, ((0, 0), (0, len_backward - len_forward), (0, 0))
            ),
            lambda: out_forward,
        )

        codes = tf.concat((out_forward, out_backward), axis=-1)
        return codes


class Encoder_6(tf.keras.layers.Layer):
    def __init__(self, hparams, **kwargs):
        super(Encoder_6, self).__init__(name='Encoder_6', **kwargs)
        self.dim_neck_3 = hparams.dim_neck_3
        self.freq_3 = hparams.freq_3
        self.dim_f0 = hparams.dim_f0
        self.dim_enc_3 = hparams.dim_enc_3
        self.dim_emb = hparams.dim_spk_emb
        self.chs_grp = hparams.chs_grp

        self.convolutions_1 = []
        for i in range(3):
            convolutions = tf.keras.Sequential()
            convolutions.add(
                ConvNorm(self.dim_enc_3, kernel_size=5, stride=1)
            )
            convolutions.add(
                GroupNormalization(groups=self.dim_enc_3 // self.chs_grp)
            )
            self.convolutions_1.append(convolutions)

        self.lstm_1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.dim_neck_3, return_sequences=True)
        )

        self.interp = InterpLnr(hparams)

    def call(self, x, training=True):
        for conv in self.convolutions_1:
            x = mish(conv(x))
            x = self.interp(
                x, tf.tile([tf.shape(x)[1]], [tf.shape(x)[0]]), training=False
            )
        outputs = self.lstm_1(x)
        out_forward = outputs[:, :, : self.dim_neck_3]
        out_backward = outputs[:, :, self.dim_neck_3:]

        codes = tf.concat(
            (
                out_forward[:, self.freq_3 - 1:: self.freq_3, :],
                out_backward[:, :: self.freq_3, :],
            ),
            axis=-1,
        )
        return codes


class Encoder_7(tf.keras.layers.Layer):
    def __init__(self, hparams, **kwargs):
        super(Encoder_7, self).__init__(name='Encoder_7', **kwargs)
        self.dim_neck = hparams.dim_neck
        self.freq = hparams.freq
        self.freq_3 = hparams.freq_3
        self.dim_enc = hparams.dim_enc
        self.dim_enc_3 = hparams.dim_enc_3
        self.dim_freq = hparams.dim_freq
        self.chs_grp = hparams.chs_grp
        self.dim_neck_3 = hparams.dim_neck_3
        self.dim_f0 = hparams.dim_f0
        self.len_org = hparams.max_len_pad

        self.convolutions_1, self.convolutions_2 = [], []
        for i in range(3):
            convolutions = tf.keras.Sequential()
            convolutions.add(
                ConvNorm(self.dim_enc, kernel_size=5, stride=1)
            )
            convolutions.add(
                GroupNormalization(groups=self.dim_enc // self.chs_grp)
            )
            self.convolutions_1.append(convolutions)

            convolutions = tf.keras.Sequential()
            convolutions.add(
                ConvNorm(self.dim_enc_3, kernel_size=5, stride=1)
            )
            convolutions.add(
                GroupNormalization(groups=self.dim_enc_3 // self.chs_grp)
            )
            self.convolutions_2.append(convolutions)

        self.lstm_1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.dim_neck, return_sequences=True)
        )
        self.lstm_2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.dim_neck_3, return_sequences=True)
        )

        self.interp = InterpLnr(hparams)

    def call(self, x_f0, training=True):
        x = x_f0[:, :, : self.dim_freq]
        f0 = x_f0[:, :, self.dim_freq:]
        for conv_1, conv_2 in zip(self.convolutions_1, self.convolutions_2):
            print(x, f0)
            x = mish(conv_1(x))
            f0 = mish(conv_2(f0))
            x_f0 = tf.concat((x, f0), axis=2)
            x_f0 = self.interp(
                x_f0,
                tf.tile([tf.shape(x_f0)[1]], [tf.shape(x)[0]]),
                training=False,
            )
            x = x_f0[:, :, : self.dim_enc]
            f0 = x_f0[:, :, self.dim_enc:]

        x = x_f0[:, :, : self.dim_enc]
        f0 = x_f0[:, :, self.dim_enc:]
        x = self.lstm_1(x)
        f0 = self.lstm_2(f0)

        x_forward = x[:, :, : self.dim_neck]
        x_backward = x[:, :, self.dim_neck:]
        f0_forward = f0[:, :, : self.dim_neck_3]
        f0_backward = f0[:, :, self.dim_neck_3:]

        codes_x = tf.concat(
            (
                x_forward[:, self.freq - 1:: self.freq, :],
                x_backward[:, :: self.freq, :],
            ),
            axis=-1,
        )
        codes_f0 = tf.concat(
            (
                f0_forward[:, self.freq_3 - 1:: self.freq_3, :],
                f0_backward[:, :: self.freq_3, :],
            ),
            axis=-1,
        )
        return codes_x, codes_f0


class Decoder_3(tf.keras.layers.Layer):
    def __init__(self, hparams, **kwargs):
        super(Decoder_3, self).__init__(name='Decoder_3', **kwargs)
        self.dim_neck = hparams.dim_neck
        self.dim_neck_2 = hparams.dim_neck_2
        self.dim_emb = hparams.dim_spk_emb
        self.dim_freq = hparams.dim_freq
        self.dim_neck_3 = hparams.dim_neck_3

        self.lstm = tf.keras.Sequential()
        for i in range(3):
            self.lstm.add(
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(512, return_sequences=True)
                )
            )

        self.linear_projection = LinearNorm(self.dim_freq)

    def call(self, x):
        outputs = self.lstm(x)
        decoder_output = self.linear_projection(outputs)
        return decoder_output


class Decoder_4(tf.keras.layers.Layer):
    def __init__(self, hparams, **kwargs):
        super(Decoder_4, self).__init__(name='Decoder_4', **kwargs)
        self.dim_neck_2 = hparams.dim_neck_2
        self.dim_f0 = hparams.dim_f0
        self.dim_neck_3 = hparams.dim_neck_3

        self.lstm = tf.keras.Sequential()
        for i in range(2):
            self.lstm.add(
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(256, return_sequences=True)
                )
            )

        self.linear_projection = LinearNorm(self.dim_f0)

    def call(self, x):
        outputs = self.lstm(x)
        decoder_output = self.linear_projection(outputs)
        return decoder_output


class Model(tf.keras.Model):
    def __init__(self, hparams, **kwargs):
        super(Model, self).__init__(name='speechsplit', **kwargs)
        self.encoder_1 = Encoder_7(hparams)
        self.encoder_2 = Encoder_t(hparams)
        self.decoder = Decoder_3(hparams)
        self.freq = hparams.freq
        self.freq_2 = hparams.freq_2
        self.freq_3 = hparams.freq_3

    def call(self, x_f0, x_org, c_trg, training=True):
        codes_x, codes_f0 = self.encoder_1(x_f0, training=training)
        codes_2 = self.encoder_2(x_org, None, training=training)
        code_exp_1 = tf.repeat(codes_x, self.freq, axis=1)
        code_exp_3 = tf.repeat(codes_f0, self.freq_3, axis=1)
        code_exp_2 = tf.repeat(codes_2, self.freq_2, axis=1)
        self.o = [code_exp_1, code_exp_2, code_exp_3, c_trg]

        c_trg = tf.tile(tf.expand_dims(c_trg, 1), (1, tf.shape(x_f0)[1], 1))

        encoder_outputs = tf.concat(
            (code_exp_1, code_exp_2, code_exp_3, c_trg), axis=-1
        )
        mel_outputs = self.decoder(encoder_outputs, training=training)

        return codes_x, codes_f0, codes_2, encoder_outputs, mel_outputs


class Model_F0(tf.keras.Model):
    def __init__(self, hparams, **kwargs):
        super(Model_F0, self).__init__(name='speechsplit_f0', **kwargs)
        self.encoder_2 = Encoder_t(hparams)
        self.encoder_3 = Encoder_6(hparams)
        self.decoder = Decoder_4(hparams)
        self.freq_2 = hparams.freq_2
        self.freq_3 = hparams.freq_3

    def call(self, x_org, f0_trg, training=True):
        codes_2 = self.encoder_2(x_org, None, training=training)
        code_exp_2 = tf.repeat(codes_2, self.freq_2, axis=1)
        codes_3 = self.encoder_3(f0_trg, training=training)
        code_exp_3 = tf.repeat(codes_3, self.freq_3, axis=1)
        self.o = [code_exp_2, code_exp_3]
        encoder_outputs = tf.concat((code_exp_2, code_exp_3), axis=-1)
        mel_outputs = self.decoder(encoder_outputs, training=training)
        return codes_2, codes_3, encoder_outputs, mel_outputs
