import tensorflow as tf
from ..fastspeech.model import TFFastSpeechEncoder
from ..utils import shape_list
import numpy as np


class Attention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(Attention, self).__init__(name = 'Attention', **kwargs)
        self.config = config
        self.encoder = TFFastSpeechEncoder(config, name = 'encoder')
        self.position_embeddings = tf.convert_to_tensor(
            self._sincos_embedding()
        )

    def call(self, x, training = True):
        lengths = tf.math.count_nonzero(x, axis = 2)
        lengths = tf.reduce_max(lengths, 1)
        max_length = tf.cast(tf.reduce_max(lengths), tf.int32)
        attention_mask = tf.sequence_mask(
            lengths = lengths, maxlen = tf.shape(x)[1], dtype = tf.float32
        )
        attention_mask.set_shape((None, None))
        seq_length = tf.shape(x)[1]
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


class MulCatBlock(tf.keras.layers.Layer):
    def __init__(self, config, input_size, dropout = 0, **kwargs):
        super(MulCatBlock, self).__init__(name = 'MulCatBlock', **kwargs)

        self.input_size = input_size

        self.rnn = Attention(config)
        self.gate_rnn = Attention(config)
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
        super(DPMulCat, self).__init__(name = 'GroupNorm', **kwargs)

    def call(self, input):
        return tf.contrib.layers.group_norm(x_tf, groups = 1, epsilon = 1e-8)


class ByPass(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ByPass, self).__init__(name = 'ByPass', **kwargs)

    def call(self, input):
        return input


class DPMulCat(tf.keras.layers.Layer):
    def __init__(
        self,
        config,
        input_size,
        output_size,
        num_spk,
        dropout = 0,
        num_layers = 1,
        input_normalize = False,
        **kwargs
    ):
        super(DPMulCat, self).__init__(name = 'DPMulCat', **kwargs)

        self.input_size = input_size
        self.output_size = output_size
        self.in_norm = input_normalize
        self.num_layers = num_layers

        self.rows_grnn = []
        self.cols_grnn = []
        self.rows_normalization = []
        self.cols_normalization = []

        for i in range(num_layers):
            self.rows_grnn.append(MulCatBlock(config, output_size, dropout))
            self.cols_grnn.append(MulCatBlock(config, output_size, dropout))
            if self.in_norm:
                self.rows_normalization.append(GroupNorm())
                self.cols_normalization.append(GroupNorm())
            else:
                # used to disable normalization
                self.rows_normalization.append(ByPass())
                self.cols_normalization.append(ByPass())

        # self.outputs = tf.keras.layers.Conv2D(
        #     output_size * num_spk, 1, padding = 'SAME'
        # )
        self.outputs = tf.keras.layers.Dense(output_size * num_spk)

    def call(self, input, training = True):
        # original, [b, d3, d1, d2],
        # [b, segment, d, t], [b, d, segment, tZ]
        input = tf.transpose(input, (0, 2, 1, 3))
        batch_size, d3, d1, d2 = shape_list(input)
        output = input
        output_all = []
        for i in range(self.num_layers):
            row_input = tf.transpose(output, [0, 3, 2, 1])
            row_input = tf.reshape(row_input, (batch_size * d2, d1, d3))
            row_output = self.rows_grnn[i](row_input, training = training)
            row_output = tf.reshape(row_output, (batch_size, d2, d1, d3))
            row_output = tf.transpose(row_output, (0, 3, 2, 1))
            row_output = self.rows_normalization[i](row_output)
            output = output + row_output

            col_input = tf.transpose(output, [0, 2, 3, 1])
            col_input = tf.reshape(col_input, (batch_size * d1, d2, d3))
            col_output = self.cols_grnn[i](col_input, training = training)
            col_output = tf.reshape(col_output, (batch_size, d1, d2, d3))
            col_output = tf.transpose(col_output, (0, 3, 1, 2))
            col_output = self.cols_normalization[i](col_output)

            output = output + col_output

            # torch.Size([1, 128, 126, 360]
            t = tf.transpose(output, [0, 2, 3, 1])
            output_i = self.outputs(t)
            output_all.append(output_i)
        return output_all


class Separator(tf.keras.layers.Layer):
    def __init__(
        self,
        config,
        input_dim,
        output_dim,
        num_spk = 2,
        layer = 4,
        segment_size = 100,
        input_normalize = False,
        **kwargs
    ):
        super(Separator, self).__init__(name = 'Separator', **kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layer = layer
        self.segment_size = segment_size
        self.num_spk = num_spk
        self.input_normalize = input_normalize

        self.rnn_model = DPMulCat(
            config,
            input_size = self.input_dim,
            output_size = self.output_dim,
            num_spk = self.num_spk,
            num_layers = layer,
            input_normalize = input_normalize,
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
            pad = tf.zeros(shape = (batch_size, rest, dim))
            i = tf.concat([input, pad], 1)
            return i

        input = tf.cond(rest > 0, f1, lambda: input)

        pad_aux = tf.zeros(shape = (batch_size, segment_stride, dim))
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
        segments = tf.concat([segments1, segments2], axis = 3)
        segments = tf.reshape(segments, (batch_size, -1, dim, segment_size))
        segments = tf.transpose(segments, perm = [0, 3, 2, 1])
        return segments, rest

    def merge_chuncks(self, input, rest):
        # original, [b, dim, segment_size, _]
        # torch.Size([1, 256, 126, 360])
        # (1, 126, 360, 256)
        input = tf.transpose(input, perm = [0, 3, 1, 2])
        batch_size, dim, segment_size, _ = shape_list(input)
        segment_stride = segment_size // 2
        # original, [b, dim, _, segment_size]
        input = tf.transpose(input, perm = [0, 1, 3, 2])
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

        return tf.transpose(output, perm = [0, 2, 1])

    def call(self, input, training = True):
        # create chunks
        enc_segments, enc_rest = self.create_chuncks(input, self.segment_size)
        output_all = self.rnn_model(enc_segments, training = training)
        output_all_wav = []
        for ii in range(len(output_all)):
            output_ii = self.merge_chuncks(output_all[ii], enc_rest)
            output_all_wav.append(output_ii)
        return output_all_wav


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
    def __init__(
        self,
        config,
        N = 128,
        O = 128,
        R = 1,
        C = 4,
        input_normalize = False,
        segment_size = 420,
        **kwargs
    ):
        super(Model, self).__init__(name = 'fast-swave', **kwargs)
        layer = R
        num_spk = C
        self.C = C
        self.N = N
        self.O = O
        self.separator = Separator(
            config = config.encoder_self_attention_params,
            input_dim = N,
            output_dim = O,
            num_spk = num_spk,
            layer = layer,
            segment_size = segment_size,
            input_normalize = input_normalize,
        )
        self.dense = tf.keras.layers.Dense(
            N, dtype = tf.float32, name = 'dense'
        )
        self.decoder = Decoder(config.decoder_self_attention_params)
        self.mel_dense = tf.keras.layers.Dense(
            units = config.num_mels, dtype = tf.float32, name = 'mel_before'
        )

    def call(self, mixture, mel_lengths, training = True):
        mixture = self.dense(mixture)
        output_all = self.separator(mixture, training = training)
        T_mix = tf.shape(mixture)[1]
        batch_size = tf.shape(mixture)[0]

        max_length = tf.cast(tf.reduce_max(mel_lengths), tf.int32)
        attention_mask = tf.sequence_mask(
            lengths = mel_lengths, maxlen = max_length, dtype = tf.float32
        )
        attention_mask.set_shape((None, None))

        outputs = []
        for ii in range(len(output_all)):
            decoder_output = self.decoder(
                output_all[ii], attention_mask, training = training
            )
            output_ii = tf.reshape(
                decoder_output, (batch_size, T_mix, self.C, self.O)
            )
            outputs.append(self.mel_dense(output_ii))
        return outputs, output_all
