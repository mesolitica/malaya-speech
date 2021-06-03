import tensorflow as tf
from ..utils import shape_list
import numpy as np


class Encoder(tf.keras.layers.Layer):
    def __init__(self, L, N, **kwargs):
        super(Encoder, self).__init__(name='Encoder', **kwargs)
        self.conv = tf.keras.layers.Conv1D(
            N, kernel_size=L, strides=L // 2, use_bias=False
        )

    def call(self, mixture):
        mixture = tf.expand_dims(mixture, -1)
        mixture_w = tf.nn.relu(self.conv(mixture))
        return mixture_w


class Decoder(tf.keras.layers.Layer):
    def __init__(self, L, **kwargs):
        super(Decoder, self).__init__(name='Decoder', **kwargs)
        self.L = L

    def call(self, est_source):
        # torch.Size([1, 256, 22521])
        # pt (1, 2, 128, 22521), tf (1, 22521, 2, 128)
        est_source = tf.transpose(est_source, (0, 1, 3, 2))
        est_source = tf.compat.v1.layers.average_pooling2d(
            est_source, 1, (1, 8), padding='SAME'
        )
        est_source = tf.signal.overlap_and_add(
            tf.transpose(est_source, (0, 3, 1, 2)), self.L // 2
        )

        return est_source


class MulCatBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        input_size,
        hidden_size,
        dropout=0,
        bidirectional=False,
        **kwargs
    ):
        super(MulCatBlock, self).__init__(name='MulCatBlock', **kwargs)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        if bidirectional:
            self.rnn = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(hidden_size, return_sequences=True)
            )
            self.gate_rnn = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(hidden_size, return_sequences=True)
            )
        else:
            self.rnn = tf.keras.layers.LSTM(
                hidden_size, return_sequences=True
            )
            self.gate_rnn = tf.keras.layers.LSTM(
                hidden_size, return_sequences=True
            )

        self.rnn_proj = tf.keras.layers.Dense(input_size)
        self.gate_rnn_proj = tf.keras.layers.Dense(input_size)
        self.block_projection = tf.keras.layers.Dense(input_size)

    def call(self, input):
        output = input
        rnn_output = self.rnn(output)
        rnn_output = self.rnn_proj(rnn_output)
        gate_rnn_output = self.gate_rnn(output)
        gate_rnn_output = self.gate_rnn_proj(gate_rnn_output)
        gated_output = tf.multiply(rnn_output, gate_rnn_output)
        gated_output = tf.concat([gated_output, output], 2)
        gated_output = self.block_projection(gated_output)
        return gated_output


class GroupNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DPMulCat, self).__init__(name='GroupNorm', **kwargs)

    def call(self, input):
        return tf.contrib.layers.group_norm(x_tf, groups=1, epsilon=1e-8)


class ByPass(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ByPass, self).__init__(name='ByPass', **kwargs)

    def call(self, input):
        return input


class DPMulCat(tf.keras.layers.Layer):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_spk,
        dropout=0,
        num_layers=1,
        bidirectional=True,
        input_normalize=False,
        **kwargs
    ):
        super(DPMulCat, self).__init__(name='DPMulCat', **kwargs)

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.in_norm = input_normalize
        self.num_layers = num_layers

        self.rows_grnn = []
        self.cols_grnn = []
        self.rows_normalization = []
        self.cols_normalization = []

        for i in range(num_layers):
            self.rows_grnn.append(
                MulCatBlock(
                    input_size,
                    hidden_size,
                    dropout,
                    bidirectional=bidirectional,
                )
            )
            self.cols_grnn.append(
                MulCatBlock(
                    input_size,
                    hidden_size,
                    dropout,
                    bidirectional=bidirectional,
                )
            )
            if self.in_norm:
                self.rows_normalization.append(GroupNorm())
                self.cols_normalization.append(GroupNorm())
            else:
                # used to disable normalization
                self.rows_normalization.append(ByPass())
                self.cols_normalization.append(ByPass())

        self.outputs = tf.keras.layers.Conv2D(
            output_size * num_spk, 1, padding='SAME'
        )

    def call(self, input):
        # original, [b, d3, d1, d2]
        input = tf.transpose(input, (0, 2, 1, 3))
        batch_size, d3, d1, d2 = shape_list(input)
        output = input
        output_all = []
        for i in range(self.num_layers):
            row_input = tf.transpose(output, [0, 3, 2, 1])
            row_input = tf.reshape(row_input, (batch_size * d2, d1, d3))
            row_output = self.rows_grnn[i](row_input)
            row_output = tf.reshape(row_output, (batch_size, d2, d1, d3))
            row_output = tf.transpose(row_output, (0, 3, 2, 1))
            row_output = self.rows_normalization[i](row_output)
            output = output + row_output

            col_input = tf.transpose(output, [0, 2, 3, 1])
            col_input = tf.reshape(col_input, (batch_size * d1, d2, d3))
            col_output = self.cols_grnn[i](col_input)
            col_output = tf.reshape(col_output, (batch_size, d1, d2, d3))
            col_output = tf.transpose(col_output, (0, 3, 1, 2))
            col_output = self.cols_normalization[i](col_output)

            output = output + col_output

            # torch.Size([1, 128, 126, 360]
            t = tf.transpose(output, [0, 2, 3, 1])
            output_i = self.outputs(
                tf.keras.layers.PReLU(shared_axes=[1, 2])(t)
            )
            output_all.append(output_i)
        return output_all


class Separator(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim,
        feature_dim,
        hidden_dim,
        output_dim,
        num_spk=2,
        layer=4,
        segment_size=100,
        input_normalize=False,
        bidirectional=True,
        **kwargs
    ):
        super(Separator, self).__init__(name='Separator', **kwargs)
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layer = layer
        self.segment_size = segment_size
        self.num_spk = num_spk
        self.input_normalize = input_normalize

        self.rnn_model = DPMulCat(
            self.feature_dim,
            self.hidden_dim,
            self.feature_dim,
            self.num_spk,
            num_layers=layer,
            bidirectional=bidirectional,
            input_normalize=input_normalize,
        )

    def pad_segment(self, input, segment_size):
        # input is the features: (B, N, T)

        batch_size, seq_len, dim = shape_list(input)
        segment_stride = segment_size // 2
        rest = (
            segment_size
            - (segment_stride + seq_len % segment_size) % segment_size
        )

        def f1():
            pad = tf.zeros(shape=(batch_size, rest, dim))
            i = tf.concat([input, pad], 1)
            return i

        input = tf.cond(rest > 0, f1, lambda: input)

        pad_aux = tf.zeros(shape=(batch_size, segment_stride, dim))
        input = tf.concat([pad_aux, input, pad_aux], 1)
        return input, rest

    def create_chuncks(self, input, segment_size):

        input, rest = self.pad_segment(input, segment_size)
        batch_size, seq_len, dim = shape_list(input)
        segment_stride = segment_size // 2
        segments1 = tf.reshape(
            input[:, :-segment_stride], (batch_size, -1, dim, segment_size)
        )
        segments2 = tf.reshape(
            input[:, segment_stride:], (batch_size, -1, dim, segment_size)
        )
        segments = tf.concat([segments1, segments2], axis=3)
        segments = tf.reshape(segments, (batch_size, -1, dim, segment_size))
        segments = tf.transpose(segments, perm=[0, 3, 2, 1])
        return segments, rest

    def merge_chuncks(self, input, rest):
        # original, [b, dim, segment_size, _]
        # torch.Size([1, 256, 126, 360])
        # (1, 126, 360, 256)
        input = tf.transpose(input, perm=[0, 3, 1, 2])
        batch_size, dim, segment_size, _ = shape_list(input)
        segment_stride = segment_size // 2
        # original, [b, dim, _, segment_size]
        input = tf.transpose(input, perm=[0, 1, 3, 2])
        input = tf.reshape(input, (batch_size, dim, -1, segment_size * 2))

        input1 = tf.reshape(
            input[:, :, :, :segment_size], (batch_size, dim, -1)
        )[:, :, segment_stride:]
        input2 = tf.reshape(
            input[:, :, :, segment_size:], (batch_size, dim, -1)
        )[:, :, :-segment_stride]

        output = input1 + input2
        if rest > 0:
            output = output[:, :, :-rest]

        return tf.transpose(output, perm=[0, 2, 1])

    def call(self, input):
        # create chunks
        enc_segments, enc_rest = self.create_chuncks(input, self.segment_size)
        output_all = self.rnn_model(enc_segments)
        output_all_wav = []
        for ii in range(len(output_all)):
            output_ii = self.merge_chuncks(output_all[ii], enc_rest)
            output_all_wav.append(output_ii)
        return output_all_wav


class Model(tf.keras.Model):
    def __init__(
        self,
        N=128,
        L=8,
        H=128,
        R=6,
        C=2,
        input_normalize=False,
        sample_rate=8000,
        segment=4,
        **kwargs
    ):
        super(Model, self).__init__(name='swave', **kwargs)
        sr = sample_rate
        context_len = 2 * sr / 1000
        context = int(sr * context_len / 1000)
        layer = R
        filter_dim = context * 2 + 1
        num_spk = C
        segment_size = int(np.sqrt(2 * sr * segment / (L / 2)))
        self.C = C
        self.N = N
        self.encoder = Encoder(L, N)
        self.separator = Separator(
            filter_dim + N,
            N,
            H,
            filter_dim,
            num_spk,
            layer,
            segment_size,
            input_normalize,
        )
        self.decoder = Decoder(L)

    def call(self, mixture):
        mixture_w = self.encoder(mixture)
        output_all = self.separator(mixture_w)
        T_mix = tf.shape(mixture)[1]
        batch_size = tf.shape(mixture)[0]
        T_mix_w = tf.shape(mixture_w)[1]
        # generate wav after each RNN block and optimize the loss

        def pad(x, l):
            return tf.pad(x, [[0, 0], [0, 0], [0]])

        def slice(x, l):
            return x[:, :, :l]

        outputs = []
        for ii in range(len(output_all)):
            output_ii = tf.reshape(
                output_all[ii], (batch_size, T_mix_w, self.C, self.N)
            )
            output_ii = self.decoder(output_ii)
            output_ii = tf.cond(
                tf.shape(output_ii)[2] >= T_mix,
                lambda: output_ii[:, :, :T_mix],
                lambda: tf.pad(
                    output_ii,
                    [[0, 0], [0, 0], [0, T_mix - tf.shape(output_ii)[2]]],
                ),
            )
            outputs.append(output_ii)
        return outputs, output_all
