import tensorflow as tf
from ..conformer.layer import PositionalEncoding
from ..fastspeech.model import TFFastSpeechEncoder
from ..utils import shape_list
import numpy as np


class Encoder(tf.keras.layers.Layer):
    def __init__(self, kernel_size = 2, out_channels = 64, **kwargs):
        super(Encoder, self).__init__(name = 'Encoder', **kwargs)
        self.conv1d = tf.keras.layers.Conv1D(
            out_channels,
            kernel_size = kernel_size,
            strides = kernel_size // 2,
            padding = 'VALID',
        )

    def call(self, x):
        x = self.conv1d(x)
        return tf.nn.relu(x)


class Dual_Computation_Block(tf.keras.layers.Layer):
    def __init__(
        self,
        intra_mdl,
        inter_mdl,
        out_channels,
        norm = 'ln',
        skip_around_intra = True,
        linear_layer_after_inter_intra = True,
        **kwargs,
    ):
        super(Dual_Computation_Block, self).__init__(
            name = 'Dual_Computation_Block', **kwargs
        )
        self.intra_mdl = intra_mdl
        self.inter_mdl = inter_mdl
        self.skip_around_intra = skip_around_intra
        self.linear_layer_after_inter_intra = linear_layer_after_inter_intra

        self.norm = norm
        if norm is not None:
            self.intra_norm = tf.keras.layers.LayerNormalization(
                epsilon = 1e-8, axis = 1
            )
            self.inter_norm = tf.keras.layers.LayerNormalization(
                epsilon = 1e-8, axis = 1
            )

        if linear_layer_after_inter_intra:
            self.intra_linear = tf.keras.layers.Dense(out_channels)
            self.inter_linear = tf.keras.layers.Dense(out_channels)

    def call(self, x, training = True):
        B, N, K, S = shape_list(x)
        intra = tf.reshape(tf.transpose(x, (0, 3, 2, 1)), (B * S, K, N))
        intra = self.intra_mdl(intra, training = training)
        if self.linear_layer_after_inter_intra:
            intra = self.intra_linear(intra)

        intra = tf.reshape(intra, (B, S, K, N))
        intra = tf.transpose(intra, (0, 3, 2, 1))
        if self.norm is not None:
            intra = self.intra_norm(intra)

        if self.skip_around_intra:
            intra = intra + x

        inter = tf.reshape(tf.transpose(x, (0, 2, 3, 1)), (B * K, S, N))
        inter = self.inter_mdl(inter, training = training)
        if self.linear_layer_after_inter_intra:
            inter = self.inter_linear(inter)

        inter = tf.reshape(inter, (B, K, S, N))
        inter = tf.transpose(inter, (0, 3, 1, 2))
        if self.norm is not None:
            inter = self.inter_norm(inter)

        out = inter + intra
        return out


class Dual_Path_Model(tf.keras.layers.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        intra_model,
        inter_model,
        num_layers = 1,
        K = 200,
        num_spks = 2,
        skip_around_intra = True,
        linear_layer_after_inter_intra = True,
        use_global_pos_enc = False,
        max_length = 20000,
        activation = tf.nn.relu,
        **kwargs,
    ):
        super(Dual_Path_Model, self).__init__(
            name = 'Dual_Path_Model', **kwargs
        )
        self.K = K
        self.num_spks = num_spks
        self.num_layers = num_layers
        self.norm = tf.keras.layers.LayerNormalization(
            axis = -1, epsilon = 1e-8
        )
        self.conv1d = tf.keras.layers.Conv1D(out_channels, 1, use_bias = False)
        self.use_global_pos_enc = use_global_pos_enc
        if self.use_global_pos_enc:
            self.pos_enc = PositionalEncoding()

        self.conv2d = tf.keras.layers.Conv2D(
            out_channels * num_spks, kernel_size = 1
        )
        self.end_conv1x1 = tf.keras.layers.Conv1D(
            in_channels, 1, use_bias = False
        )
        self.prelu = tf.keras.layers.PReLU(shared_axes = [2, 3])
        self.output_left = tf.keras.layers.Conv1D(out_channels, 1)
        self.output_gate = tf.keras.layers.Conv1D(out_channels, 1)
        self.activation = activation

        self.dual_mdl = []
        for i in range(num_layers):
            self.dual_mdl.append(
                Dual_Computation_Block(
                    intra_model(),
                    inter_model(),
                    out_channels,
                    skip_around_intra = skip_around_intra,
                    linear_layer_after_inter_intra = linear_layer_after_inter_intra,
                )
            )

        self.out_channels = out_channels

    def call(self, x, training = True):
        # B, L, N
        B, L, N = shape_list(x)
        x = self.norm(x)
        x = self.conv1d(x)
        if self.use_global_pos_enc:
            x = self.pos_enc(x) + x * (N ** 0.5)

        # B, N, L
        x = tf.transpose(x, (0, 2, 1))
        # B, N, K, S
        x, gap = self._Segmentation(x, self.K)
        # B, N, K, S
        for i in range(self.num_layers):
            x = self.dual_mdl[i](x, training = training)
        x = self.prelu(x)

        # B, K, S, N
        x = tf.transpose(x, (0, 2, 3, 1))
        # B, K, S, N*spks
        x = self.conv2d(x)

        # B, N*spks, K, S
        x = tf.transpose(x, (0, 3, 1, 2))
        B, _, K, S = shape_list(x)

        # B*spks, N, L
        x = tf.reshape(x, (B * self.num_spks, -1, K, S))
        x = self._over_add(x, gap)

        # B*spks, L, N
        x = tf.transpose(x, (0, 2, 1))
        x.set_shape((None, None, self.out_channels))
        x = tf.nn.tanh(self.output_left(x)) + tf.nn.sigmoid(self.output_gate(x))
        x = self.end_conv1x1(x)

        _, L, N = shape_list(x)
        x = tf.reshape(x, (B, self.num_spks, L, N))
        if self.activation:
            x = self.activation(x)
        # spks, B, N, L
        return tf.transpose(x, (1, 0, 2, 3))

    def _padding(self, input, K):
        B, N, L = shape_list(input)
        P = K // 2
        gap = K - (P + L % K) % K

        def f1():
            pad = tf.zeros(shape = (B, N, gap))
            i = tf.concat([input, pad], axis = 2)
            return i

        input = tf.cond(gap > 0, f1, lambda: input)

        _pad = tf.zeros(shape = (B, N, P))
        input = tf.concat([_pad, input, _pad], axis = 2)
        return input, gap

    def _Segmentation(self, input, K):
        B, N, L = shape_list(input)
        P = K // 2
        input, gap = self._padding(input, K)
        input1 = tf.reshape(input[:, :, :-P], (B, N, -1, K))
        input2 = tf.reshape(input[:, :, P:], (B, N, -1, K))
        input = tf.concat([input1, input2], axis = 3)
        input = tf.reshape(input, (B, N, -1, K))
        input = tf.transpose(input, (0, 1, 3, 2))
        return input, gap

    def _over_add(self, input, gap):
        B, N, K, S = shape_list(input)
        P = K // 2
        input = tf.reshape(tf.transpose(input, (0, 1, 3, 2)), (B, N, -1, K * 2))
        input1 = tf.reshape(input[:, :, :, :K], (B, N, -1))[:, :, P:]
        input2 = tf.reshape(input[:, :, :, K:], (B, N, -1))[:, :, :-P]
        input = input1 + input2

        input = tf.cond(gap > 0, lambda: input[:, :, :-gap], lambda: input)
        return input


class Conv1DTranspose(tf.keras.layers.Layer):
    def __init__(
        self, filters, kernel_size, strides, activation, use_bias, **kwargs
    ):
        super(Conv1DTranspose, self).__init__(
            name = 'Conv1DTranspose', **kwargs
        )
        self.conv = tf.keras.layers.Conv2DTranspose(
            filters,
            (kernel_size, 1),
            strides = (strides, 1),
            activation = activation,
            use_bias = use_bias,
        )

    def call(self, x):
        x = tf.expand_dims(x, 2)
        x = self.conv(x)
        return x[:, :, 0]


class Encoder_FastSpeech(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(Encoder_FastSpeech, self).__init__(
            name = 'Encoder_FastSpeech', **kwargs
        )
        self.config = config
        self.encoder = TFFastSpeechEncoder(config, name = 'encoder')
        self.pos_enc = PositionalEncoding()

    def call(self, x, lengths = None, training = True):
        if lengths is None:
            lengths = tf.tile([tf.shape(x)[1]], [tf.shape(x)[0]])
        max_length = tf.cast(tf.reduce_max(lengths), tf.int32)
        attention_mask = tf.sequence_mask(
            lengths = lengths, maxlen = max_length, dtype = tf.float32
        )
        attention_mask.set_shape((None, None))
        x = self.pos_enc(x) + x
        f = self.encoder([x, attention_mask], training = training)[0]
        return f


class Model(tf.keras.Model):
    def __init__(
        self,
        intra_model,
        inter_model,
        encoder_kernel_size = 16,
        encoder_in_nchannels = 1,
        encoder_out_nchannels = 256,
        masknet_chunksize = 250,
        masknet_numlayers = 2,
        masknet_useextralinearlayer = False,
        masknet_extraskipconnection = True,
        masknet_numspks = 4,
        activation = tf.nn.relu,
        **kwargs,
    ):
        super(Model, self).__init__(name = 'sepformer', **kwargs)
        self.encoder = Encoder(
            kernel_size = encoder_kernel_size,
            out_channels = encoder_out_nchannels,
        )
        self.masknet = Dual_Path_Model(
            in_channels = encoder_out_nchannels,
            out_channels = encoder_out_nchannels,
            intra_model = intra_model,
            inter_model = inter_model,
            num_layers = masknet_numlayers,
            K = masknet_chunksize,
            num_spks = masknet_numspks,
            skip_around_intra = masknet_extraskipconnection,
            linear_layer_after_inter_intra = masknet_useextralinearlayer,
            activation = activation,
        )
        self.decoder = Conv1DTranspose(
            filters = encoder_in_nchannels,
            kernel_size = encoder_kernel_size,
            strides = encoder_kernel_size // 2,
            activation = None,
            use_bias = False,
        )

        self.num_spks = masknet_numspks

    def call(self, mix, training = True):
        T_origin = tf.shape(mix)[1]
        mix_w = self.encoder(mix)
        est_mask = self.masknet(mix_w, training = training)
        mix_w = tf.tile(tf.expand_dims(mix_w, 0), (self.num_spks, 1, 1, 1))
        sep_h = mix_w * est_mask

        o = []
        for i in range(self.num_spks):
            o.append(tf.expand_dims(self.decoder(sep_h[i]), 0))

        o = tf.concat(o, axis = 0)
        T_est = tf.shape(o)[2]
        o = tf.cond(
            T_origin > T_est,
            lambda: tf.pad(o, ((0, 0), (0, 0), (0, T_origin - T_est), (0, 0))),
            lambda: o[:, :, :T_origin],
        )
        return o


class Model_Mel(tf.keras.Model):
    def __init__(
        self,
        intra_model,
        inter_model,
        decoder,
        encoder_kernel_size = 3,
        encoder_in_nchannels = 80,
        encoder_out_nchannels = 256,
        masknet_chunksize = 250,
        masknet_numlayers = 2,
        masknet_useextralinearlayer = False,
        masknet_extraskipconnection = True,
        masknet_numspks = 4,
        activation = tf.nn.relu,
        **kwargs,
    ):
        super(Model_Mel, self).__init__(name = 'sepformer', **kwargs)
        self.mel_dense = tf.keras.layers.Dense(
            units = encoder_in_nchannels,
            dtype = tf.float32,
            name = 'mel_before',
        )
        self.encoder = tf.keras.layers.Conv1D(
            encoder_out_nchannels,
            kernel_size = encoder_kernel_size,
            strides = 1,
            padding = 'SAME',
        )
        self.masknet = Dual_Path_Model(
            in_channels = encoder_out_nchannels,
            out_channels = encoder_out_nchannels,
            intra_model = intra_model,
            inter_model = inter_model,
            num_layers = masknet_numlayers,
            K = masknet_chunksize,
            num_spks = masknet_numspks,
            skip_around_intra = masknet_extraskipconnection,
            linear_layer_after_inter_intra = masknet_useextralinearlayer,
            activation = activation,
        )
        self.decoder = decoder()

        self.num_spks = masknet_numspks

    def call(self, mix, mix_len, training = True):
        T_origin = tf.shape(mix)[1]
        mix_w = self.encoder(mix)
        est_mask = self.masknet(mix_w, training = training)
        mix_w = tf.tile(tf.expand_dims(mix_w, 0), (self.num_spks, 1, 1, 1))
        sep_h = mix_w * est_mask

        max_length = tf.cast(tf.reduce_max(mix_len), tf.int32)
        attention_mask = tf.sequence_mask(
            lengths = mix_len, maxlen = max_length, dtype = tf.float32
        )
        attention_mask.set_shape((None, None))

        o = []
        for i in range(self.num_spks):
            d = self.decoder(sep_h[i], attention_mask, training = training)
            o.append(tf.expand_dims(self.mel_dense(d), 0))

        o = tf.concat(o, axis = 0)
        return o
