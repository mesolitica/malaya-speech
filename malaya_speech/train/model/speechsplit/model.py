import tensorflow as tf
import numpy as np
from ..autovc.model import Postnet
from ..utils import GroupNormalization


def quantize_f0_tf(x, num_bins = 256):
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
    def __init__(self, out_dim, bias = True, **kwargs):
        super(LinearNorm, self).__init__(name = 'LinearNorm', **kwargs)
        self.linear_layer = tf.keras.layers.Dense(out_dim, use_bias = bias)

    def call(self, x):
        return self.linear_layer(x)


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


class GroupNorm(tf.keras.layers.Layer):
    def __init__(self, group, **kwargs):
        super(GroupNorm, self).__init__(name = 'GroupNorm', **kwargs)
        self.group = group

    def call(self, input):
        # input = NWC, [B, T, D],
        return tf.contrib.layers.group_norm(
            input,
            groups = self.group,
            epsilon = 1e-8,
            channels_axis = -1,
            reduction_axes = (-2,),
        )


class InterpLnr(tf.keras.layers.Layer):
    def __init__(self, hparams, **kwargs):
        super().__init__(**kwargs)
        self.max_len_seq = hparams.max_len_seq
        self.max_len_pad = hparams.max_len_pad

        self.min_len_seg = hparams.min_len_seg
        self.max_len_seg = hparams.max_len_seg

        self.max_num_seg = self.max_len_seq // self.min_len_seg + 1

    def call(self, x, len_seq, training = True):

        if not training:
            return x

        batch_size = tf.shape(x)[0]
        dim = x.shape[2]
        indices = tf.tile(
            tf.expand_dims(tf.range(self.max_len_seg * 2), 0),
            (batch_size * self.max_num_seg, 1),
        )
        scales = (
            tf.random.uniform(shape = (batch_size * self.max_num_seg,)) + 0.5
        )
        idx_scaled = tf.cast(indices, tf.float32) / tf.expand_dims(scales, -1)
        idx_scaled_fl = tf.math.floor(idx_scaled)
        lambda_ = idx_scaled - idx_scaled_fl
        len_seg = tf.random.uniform(
            (batch_size * self.max_num_seg, 1),
            minval = self.min_len_seg,
            maxval = self.max_len_seg,
            dtype = tf.int32,
        )
        idx_mask = idx_scaled_fl < (tf.cast(len_seg, tf.float32) - 1)
        offset = tf.math.cumsum(
            tf.reshape(len_seg, (batch_size, -1)), axis = -1
        )
        offset = tf.reshape(tf.pad(offset[:, :-1], ((0, 0), (1, 0))), (-1, 1))
        idx_scaled_org = idx_scaled_fl + tf.cast(offset, tf.float32)

        len_seq_rp = tf.repeat(len_seq, self.max_num_seg)
        idx_mask_org = idx_scaled_org < tf.cast(
            tf.expand_dims(len_seq_rp - 1, -1), tf.float32
        )

        idx_mask_final = tf.cast(idx_mask & idx_mask_org, tf.int32)

        counts = tf.reduce_sum(
            tf.reshape(
                tf.reduce_sum(idx_mask_final, axis = -1), (batch_size, -1)
            ),
            axis = -1,
        )

        index_1 = tf.repeat(tf.range(batch_size), counts)

        index_2_fl = tf.cast(
            tf.boolean_mask(idx_scaled_org, idx_mask_final), tf.int32
        )
        index_2_cl = index_2_fl + 1
        concatenated = tf.transpose(
            tf.concat(
                [
                    tf.expand_dims(index_1, axis = 0),
                    tf.expand_dims(index_2_fl, axis = 0),
                ],
                axis = 0,
            )
        )
        y_fl = tf.cast(tf.gather_nd(x, concatenated), tf.float32)

        concatenated = tf.transpose(
            tf.concat(
                [
                    tf.expand_dims(index_1, axis = 0),
                    tf.expand_dims(index_2_cl, axis = 0),
                ],
                axis = 0,
            )
        )
        y_cl = tf.cast(tf.gather_nd(x, concatenated), tf.float32)
        lambda_f = tf.expand_dims(tf.boolean_mask(lambda_, idx_mask_final), -1)

        y = (1 - lambda_f) * y_fl + lambda_f * y_cl
        sequences = tf.reshape(
            y[: (tf.shape(y)[0] // batch_size) * batch_size],
            (batch_size, -1, dim),
        )
        return tf.pad(
            sequences,
            ((0, 0), (0, self.max_len_pad - tf.shape(sequences)[1]), (0, 0)),
        )


class Encoder_t(tf.keras.layers.Layer):
    def __init__(self, hparams, **kwargs):
        super(Encoder_t, self).__init__(name = 'Encoder_t', **kwargs)
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
                ConvNorm(self.dim_enc_2, kernel_size = 5, stride = 1)
            )
            convolutions.add(
                GroupNormalization(groups = self.dim_enc_2 // self.chs_grp)
            )
            self.convolutions.append(convolutions)

        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.dim_neck_2, return_sequences = True)
        )

    def call(self, x, mask):

        for conv in self.convolutions:
            x = tf.nn.tanh(conv(x))

        outputs = self.lstm(x)
        print(outputs.shape)
        if mask is not None:
            outputs = outputs * mask

        out_forward = outputs[:, :, : self.dim_neck_2]
        out_backward = outputs[:, :, self.dim_neck_2 :]

        codes = tf.concat(
            (
                out_forward[:, self.freq_2 - 1 :: self.freq_2, :],
                out_backward[:, :: self.freq_2, :],
            ),
            axis = -1,
        )
        return codes


class Encoder_7(tf.keras.layers.Layer):
    def __init__(self, hparams, **kwargs):
        super(Encoder_7, self).__init__(name = 'Encoder_7', **kwargs)
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
                ConvNorm(self.dim_enc, kernel_size = 5, stride = 1)
            )
            convolutions.add(
                GroupNormalization(groups = self.dim_enc // self.chs_grp)
            )
            self.convolutions_1.append(convolutions)

            convolutions = tf.keras.Sequential()
            convolutions.add(
                ConvNorm(self.dim_enc_3, kernel_size = 5, stride = 1)
            )
            convolutions.add(
                GroupNormalization(groups = self.dim_enc_3 // self.chs_grp)
            )
            self.convolutions_2.append(convolutions)

        self.lstm_1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.dim_neck, return_sequences = True)
        )
        self.lstm_2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.dim_neck_3, return_sequences = True)
        )

        self.interp = InterpLnr(hparams)

    def call(self, x_f0):
        x = x_f0[:, :, : self.dim_freq]
        f0 = x_f0[:, :, self.dim_freq :]
        for conv_1, conv_2 in zip(self.convolutions_1, self.convolutions_2):
            print(x, f0)
            x = tf.nn.tanh(conv_1(x))
            f0 = tf.nn.tanh(conv_2(f0))
            x_f0 = tf.concat((x, f0), axis = 2)
            x_f0 = self.interp(x_f0, tf.tile([self.len_org], [tf.shape(x)[0]]))
            x = x_f0[:, :, : self.dim_enc]
            f0 = x_f0[:, :, self.dim_enc :]

        x = x_f0[:, :, : self.dim_enc]
        f0 = x_f0[:, :, self.dim_enc :]
        x = self.lstm_1(x)
        f0 = self.lstm_2(f0)

        x_forward = x[:, :, : self.dim_neck]
        x_backward = x[:, :, self.dim_neck :]
        f0_forward = f0[:, :, : self.dim_neck_3]
        f0_backward = f0[:, :, self.dim_neck_3 :]

        codes_x = tf.concat(
            (
                x_forward[:, self.freq - 1 :: self.freq, :],
                x_backward[:, :: self.freq, :],
            ),
            axis = -1,
        )
        codes_f0 = tf.concat(
            (
                f0_forward[:, self.freq_3 - 1 :: self.freq_3, :],
                f0_backward[:, :: self.freq_3, :],
            ),
            axis = -1,
        )
        return codes_x, codes_f0


class InterpLnr(tf.keras.layers.Layer):
    def __init__(self, hparams, **kwargs):
        super().__init__(**kwargs)
        self.max_len_seq = hparams.max_len_seq
        self.max_len_pad = hparams.max_len_pad

        self.min_len_seg = hparams.min_len_seg
        self.max_len_seg = hparams.max_len_seg

        self.max_num_seg = self.max_len_seq // self.min_len_seg + 1

    def call(self, x, len_seq, training = True):

        if not training:
            return x

        batch_size = tf.shape(x)[0]
        dim = x.shape[2]
        indices = tf.tile(
            tf.expand_dims(tf.range(self.max_len_seg * 2), 0),
            (batch_size * self.max_num_seg, 1),
        )
        scales = (
            tf.random.uniform(shape = (batch_size * self.max_num_seg,)) + 0.5
        )
        idx_scaled = tf.cast(indices, tf.float32) / tf.expand_dims(scales, -1)
        idx_scaled_fl = tf.math.floor(idx_scaled)
        lambda_ = idx_scaled - idx_scaled_fl
        len_seg = tf.random.uniform(
            (batch_size * self.max_num_seg, 1),
            minval = self.min_len_seg,
            maxval = self.max_len_seg,
            dtype = tf.int32,
        )
        idx_mask = idx_scaled_fl < (tf.cast(len_seg, tf.float32) - 1)
        offset = tf.math.cumsum(
            tf.reshape(len_seg, (batch_size, -1)), axis = -1
        )
        offset = tf.reshape(tf.pad(offset[:, :-1], ((0, 0), (1, 0))), (-1, 1))
        idx_scaled_org = idx_scaled_fl + tf.cast(offset, tf.float32)

        len_seq_rp = tf.repeat(len_seq, self.max_num_seg)
        idx_mask_org = idx_scaled_org < tf.cast(
            tf.expand_dims(len_seq_rp - 1, -1), tf.float32
        )

        idx_mask_final = tf.cast(idx_mask & idx_mask_org, tf.int32)

        counts = tf.reduce_sum(
            tf.reshape(
                tf.reduce_sum(idx_mask_final, axis = -1), (batch_size, -1)
            ),
            axis = -1,
        )

        index_1 = tf.repeat(tf.range(batch_size), counts)

        index_2_fl = tf.cast(
            tf.boolean_mask(idx_scaled_org, idx_mask_final), tf.int32
        )
        index_2_cl = index_2_fl + 1
        concatenated = tf.transpose(
            tf.concat(
                [
                    tf.expand_dims(index_1, axis = 0),
                    tf.expand_dims(index_2_fl, axis = 0),
                ],
                axis = 0,
            )
        )
        y_fl = tf.cast(tf.gather_nd(x, concatenated), tf.float32)

        concatenated = tf.transpose(
            tf.concat(
                [
                    tf.expand_dims(index_1, axis = 0),
                    tf.expand_dims(index_2_cl, axis = 0),
                ],
                axis = 0,
            )
        )
        y_cl = tf.cast(tf.gather_nd(x, concatenated), tf.float32)
        lambda_f = tf.expand_dims(tf.boolean_mask(lambda_, idx_mask_final), -1)

        y = (1 - lambda_f) * y_fl + lambda_f * y_cl
        sequences = tf.reshape(
            y[: (tf.shape(y)[0] // batch_size) * batch_size],
            (batch_size, -1, dim),
        )
        sequences = sequences[:, : self.max_len_pad]
        return tf.pad(
            sequences,
            ((0, 0), (0, self.max_len_pad - tf.shape(sequences)[1]), (0, 0)),
        )


class Decoder_3(tf.keras.layers.Layer):
    def __init__(self, hparams, **kwargs):
        super(Decoder_3, self).__init__(name = 'Decoder_3', **kwargs)
        self.dim_neck = hparams.dim_neck
        self.dim_neck_2 = hparams.dim_neck_2
        self.dim_emb = hparams.dim_spk_emb
        self.dim_freq = hparams.dim_freq
        self.dim_neck_3 = hparams.dim_neck_3

        self.lstm = tf.keras.Sequential()
        for i in range(3):
            self.lstm.add(
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(512, return_sequences = True)
                )
            )

        self.linear_projection = LinearNorm(80)

    def call(self, x):
        outputs = self.lstm(x)
        decoder_output = self.linear_projection(outputs)
        return decoder_output


class Model(tf.keras.Model):
    def __init__(self, hparams, **kwargs):
        super(Model, self).__init__(name = 'speechsplit', **kwargs)
        self.encoder_1 = Encoder_7(hparams)
        self.encoder_2 = Encoder_t(hparams)
        self.decoder = Decoder_3(hparams)
        self.freq = hparams.freq
        self.freq_2 = hparams.freq_2
        self.freq_3 = hparams.freq_3

    def call(self, x_f0, x_org, c_trg):
        codes_x, codes_f0 = self.encoder_1(x_f0)
        codes_2 = self.encoder_2(x_org, None)
        code_exp_1 = tf.repeat(codes_x, self.freq, axis = 1)
        code_exp_3 = tf.repeat(codes_f0, self.freq_3, axis = 1)
        code_exp_2 = tf.repeat(codes_2, self.freq_2, axis = 1)

        c_trg = tf.tile(tf.expand_dims(c_trg, 1), (1, tf.shape(x_f0)[1], 1))

        encoder_outputs = tf.concat(
            (code_exp_1, code_exp_2, code_exp_3, c_trg), axis = -1
        )
        mel_outputs = self.decoder(encoder_outputs)

        return codes_x, codes_f0, codes_2, encoder_outputs, mel_outputs
