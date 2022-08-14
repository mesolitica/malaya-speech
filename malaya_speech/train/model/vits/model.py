from typing import Dict, Optional, Tuple

import tensorflow.compat.v1 as tf
import numpy as np
import math

from . import commons, modules, attentions
from ..melgan.layer import WeightNormalization, GroupConv1D
from ..melgan.model import TFReflectionPad1d
from ..utils import shape_list


class StochasticDurationPredictor(tf.keras.layers.Layer):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0, **kwargs):
        super(StochasticDurationPredictor, self).__init__(**kwargs)
        filter_channels = in_channels  # it needs to be removed from future version.
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = modules.Log()
        self.flows = []
        self.flows.append(modules.ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.flows.append(modules.Flip())

        self.post_pre = tf.keras.layers.Conv1D(filter_channels, 1, padding='SAME')
        self.post_proj = tf.keras.layers.Conv1D(filter_channels, 1, padding='SAME')
        self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)

        self.post_flows = []
        self.post_flows.append(modules.ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.post_flows.append(modules.Flip())

        self.pre = tf.keras.layers.Conv1D(filter_channels, 1, padding='SAME')
        self.proj = tf.keras.layers.Conv1D(filter_channels, 1, padding='SAME')
        self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        if gin_channels != 0:
            self.cond = tf.keras.layers.Conv1D(filter_channels, 1, padding='SAME')

    def call(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0, training=True):
        x = self.pre(x)
        if g is not None:
            x = x + self.cond(tf.stop_gradient(g))
        x = self.convs(x, x_mask, training=training)
        x = self.proj(x) * x_mask
        if not reverse:
            # original is [b, 1, t_t]
            b, t_t, _ = shape_list(w)
            flows = self.flows
            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask, training=training)
            h_w = self.post_proj(h_w) * x_mask
            e_q = tf.random.normal(shape=(b, t_t, 2)) * x_mask
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w), training=training)
                logdet_tot_q += logdet_q
            z_u, z1 = z_q[:, :, :1], z_q[:, :, 1:]
            u = tf.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += tf.reduce_sum((tf.math.log_sigmoid(z_u) + tf.math.log_sigmoid(-z_u)) * x_mask, axis=[2, 1])

            logq = tf.reduce_sum(-0.5 * (math.log(2*math.pi) + (e_q**2)) * x_mask, axis=[2, 1]) - logdet_tot_q

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = tf.concat([z0, z1], 2)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse, training=training)
                logdet_tot = logdet_tot + logdet

            nll = tf.reduce_sum(0.5 * (math.log(2*math.pi) + (z**2)) * x_mask, axis=[2, 1]) - logdet_tot
            return nll + logq
        else:
            b, t_t, _ = shape_list(x)
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]
            z = tf.random.normal(shape=(b, t_t, 2)) * noise_scale
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse, training=training)
            z0, z1 = z[:, :, :1], z[:, :, 1:]
            logw = z0
            return logw


class DurationPredictor(tf.keras.layers.Layer):
    def __init__(self, filter_channels, kernel_size, p_dropout, gin_channels=0, **kwargs):
        super(DurationPredictor, self).__init__(**kwargs)

        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = tf.keras.layers.Dropout(p_dropout)
        self.conv_1 = tf.keras.layers.Conv1D(filter_channels, kernel_size, padding='SAME')
        self.norm_1 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)
        self.conv_2 = tf.keras.layers.Conv1D(filter_channels, kernel_size, padding='SAME')
        self.norm_2 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)
        self.proj = tf.keras.layers.Conv1D(1, 1)

        if gin_channels != 0:
            self.cond = tf.keras.layers.Conv1D(in_channels, 1, padding='SAME')

    def call(self, x, x_mask, g=None, training=True):
        if g is not None:
            x = x + self.cond(tf.stop_gradient(g))
        x = self.conv_1(x * x_mask)
        x = tf.nn.relu6(x)
        x = self.norm_1(x, training=training)
        x = self.drop(x, training=training)
        x = self.conv_2(x * x_mask)
        x = tf.nn.relu6(x)
        x = self.norm_2(x, training=training)
        x = self.drop(x, training=training)
        x = self.proj(x * x_mask)
        return x * x_mask


class TextEncoder(tf.keras.layers.Layer):
    def __init__(self, n_vocab,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout, **kwargs):
        super(TextEncoder, self).__init__(**kwargs)

        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = tf.keras.layers.Embedding(n_vocab, hidden_channels)
        self.encoder = attentions.Encoder(hidden_channels,
                                          filter_channels,
                                          n_heads,
                                          n_layers,
                                          kernel_size,
                                          p_dropout)
        self.proj = tf.keras.layers.Conv1D(out_channels * 2, 1, padding='SAME')

    def call(self, x, x_lengths, training=True):
        x = self.emb(x) * math.sqrt(self.hidden_channels)
        x_mask = tf.expand_dims(commons.sequence_mask(x_lengths, tf.shape(x)[1]), 2)
        x = self.encoder(x * x_mask, x_mask, training=training)
        stats = self.proj(x) * x_mask

        m, logs = stats[:, :, :self.out_channels], stats[:, :, self.out_channels:]
        return x, m, logs, x_mask


class ResidualCouplingBlock(tf.keras.layers.Layer):
    def __init__(self, channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 n_flows=4,
                 gin_channels=0, **kwargs):
        super(ResidualCouplingBlock, self).__init__(**kwargs)
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = []
        for i in range(n_flows):
            self.flows.append(modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size,
                              dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
            self.flows.append(modules.Flip())

    def call(self, x, x_mask, g=None, reverse=False, training=True):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse, training=training)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse, training=training)
        return x


class PosteriorEncoder(tf.keras.layers.Layer):
    def __init__(self, in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0, **kwargs):
        super(PosteriorEncoder, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = tf.keras.layers.Conv1D(hidden_channels, 1, padding='SAME')
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = tf.keras.layers.Conv1D(out_channels * 2, 1, padding='SAME')

    def call(self, x, x_lengths, g=None, training=True):
        x_mask = tf.expand_dims(commons.sequence_mask(x_lengths, tf.shape(x)[1]), 2)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g, training=training)
        stats = self.proj(x) * x_mask
        m, logs = stats[:, :, :self.out_channels], stats[:, :, self.out_channels:]
        z = (m + tf.random.normal(shape=tf.shape(m)) * tf.exp(logs)) * x_mask
        return z, m, logs, x_mask


class TFConvTranspose1d(tf.keras.layers.Layer):
    """Tensorflow ConvTranspose1d module."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides,
        padding,
        is_weight_norm,
        initializer,
        **kwargs
    ):
        """Initialize TFConvTranspose1d( module.
        Args:
            filters (int): Number of filters.
            kernel_size (int): kernel size.
            strides (int): Stride width.
            padding (str): Padding type ("same" or "valid").
        """
        super().__init__(**kwargs)
        self.conv1d_transpose = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=(kernel_size, 1),
            strides=(strides, 1),
            padding='same',
            kernel_initializer=initializer,
        )
        if is_weight_norm:
            self.conv1d_transpose = WeightNormalization(self.conv1d_transpose)

    def call(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, T, C).
        Returns:
            Tensor: Output tensor (B, T', C').
        """
        x = tf.expand_dims(x, 2)
        x = self.conv1d_transpose(x)
        x = tf.squeeze(x, 2)
        return x


class Generator(tf.keras.layers.Layer):
    def __init__(
            self,
            initial_channel,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=0,
            **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        kernel_size = 7
        self.early_pad = TFReflectionPad1d(
            (kernel_size - 1) // 2,
            padding_type='REFLECT',
            name='last_reflect_padding',
        )
        self.conv_pre = tf.keras.layers.Conv1D(upsample_initial_channel, kernel_size, 1)
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        initializer = tf.keras.initializers.RandomNormal(
            mean=0.0, stddev=0.01, seed=None
        )

        self.ups = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(TFConvTranspose1d(
                filters=upsample_initial_channel//(2**(i+1)),
                kernel_size=k,
                strides=u,
                padding='same',
                is_weight_norm=True,
                initializer=initializer,
                name='conv_transpose_._{}'.format(i),
            ))

        self.resblocks = []
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.last_pad = TFReflectionPad1d(
            (kernel_size - 1) // 2,
            padding_type='REFLECT',
            name='last_reflect_padding',
        )

        self.conv_post = tf.keras.layers.Conv1D(1, kernel_size, 1, use_bias=False)

        if gin_channels != 0:
            self.cond = tf.keras.layers.Conv1D(upsample_initial_channel, 1, padding='same')

    def call(self, x, g=None, training=True):
        x = self.conv_pre(self.early_pad(x))
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = tf.keras.layers.LeakyReLU(alpha=modules.LRELU_SLOPE)(x)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels

        x = tf.keras.layers.LeakyReLU()(x)
        x = self.conv_post(self.last_pad(x))
        x = tf.tanh(x)

        return x


class DiscriminatorP(tf.keras.layers.Layer):
    def __init__(self, period, kernel_size=5, stride=3, **kwargs):
        super(DiscriminatorP, self).__init__(**kwargs)
        self.period = period
        norm_f = WeightNormalization

        self.convs = [
            norm_f(tf.keras.layers.Conv2D(32, (kernel_size, 1), (stride, 1), padding='SAME')),
            norm_f(tf.keras.layers.Conv2D(128, (kernel_size, 1), (stride, 1), padding='SAME')),
            norm_f(tf.keras.layers.Conv2D(512, (kernel_size, 1), (stride, 1), padding='SAME')),
            norm_f(tf.keras.layers.Conv2D(1024, (kernel_size, 1), (stride, 1), padding='SAME')),
            norm_f(tf.keras.layers.Conv2D(1024, (kernel_size, 1), 1, padding='SAME')),
        ]
        self.conv_post = norm_f(tf.keras.layers.Conv2D(1, (3, 1), 1, padding='SAME'))

    def call(self, x):
        fmap = []
        b, t, c = shape_list(x)

        def f1():
            n_pad = self.period - (t % self.period)
            x_ = tf.pad(x, [[0, 0], [0, n_pad], [0, 0]])
            return x_

        x = tf.cond(tf.math.not_equal(t % self.period, 0), f1, lambda: x)
        t = tf.shape(x)[1]
        x = tf.reshape(x, [b, t // self.period, self.period, c])

        for l in self.convs:
            x = l(x)
            x = tf.keras.layers.LeakyReLU(alpha=modules.LRELU_SLOPE)(x)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        x = tf.reshape(x, [b, -1])

        return x, fmap


class DiscriminatorS(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DiscriminatorS, self).__init__(**kwargs)
        norm_f = WeightNormalization

        self.convs = [norm_f(tf.keras.layers.Conv1D(16, 15, 1, padding='SAME'))]
        with tf.keras.utils.CustomObjectScope({'GroupConv1D': GroupConv1D}):
            self.convs.extend(
                [
                    norm_f(GroupConv1D(
                        filters=64,
                        kernel_size=41,
                        strides=4,
                        padding='same',
                        groups=4,
                    )),
                    norm_f(GroupConv1D(
                        filters=256,
                        kernel_size=41,
                        strides=4,
                        padding='same',
                        groups=16,
                    )),
                    norm_f(GroupConv1D(
                        filters=1024,
                        kernel_size=41,
                        strides=4,
                        padding='same',
                        groups=64,
                    )),
                    norm_f(GroupConv1D(
                        filters=1024,
                        kernel_size=41,
                        strides=4,
                        padding='same',
                        groups=256,
                    ))
                ]
            )
        self.convs.append(norm_f(tf.keras.layers.Conv1D(1024, 5, 1, padding='SAME')))
        self.conv_post = norm_f(tf.keras.layers.Conv1D(1, 3, 1, padding='SAME'))

    def call(self, x):
        fmap = []
        b, t, c = shape_list(x)

        for l in self.convs:
            x = l(x)
            x = tf.keras.layers.LeakyReLU(alpha=modules.LRELU_SLOPE)(x)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        x = tf.reshape(x, [b, -1])

        return x, fmap


class MultiPeriodDiscriminator(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MultiPeriodDiscriminator, self).__init__(**kwargs)
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS()]
        discs = discs + [DiscriminatorP(i) for i in periods]
        self.discriminators = discs

    def call(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class Model(tf.keras.Model):

    def __init__(self,
                 n_vocab,
                 spec_channels,
                 segment_size,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 n_speakers=0,
                 gin_channels=0,
                 use_sdp=True,
                 prob_predictor=0.1,
                 **kwargs):
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels

        self.use_sdp = use_sdp

        self.enc_p = TextEncoder(n_vocab,
                                 inter_channels,
                                 hidden_channels,
                                 filter_channels,
                                 n_heads,
                                 n_layers,
                                 kernel_size,
                                 p_dropout)
        self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes,
                             upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels,
                                      5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

        if use_sdp:
            print('using StochasticDurationPredictor')
            self.dp = StochasticDurationPredictor(hidden_channels, 192, 3, prob_predictor, 4, gin_channels=gin_channels)
        else:
            self.dp = DurationPredictor(256, 3, prob_predictor, gin_channels=gin_channels)

        if n_speakers > 1:
            self.emb_g = tf.keras.layers.Embedding(n_speakers, gin_channels)

    def call(self, x, x_lengths, y, y_lengths, sid=None, training=True):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, training=training)
        if self.n_speakers > 0:
            g = tf.expand_dims(self.emb_g(sid), -1)
        else:
            g = None

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g, training=training)
        z_p = self.flow(z, y_mask, g=g, training=training)

        s_p_sq_r = tf.exp(-2 * logs_p)
        neg_cent1 = tf.reduce_sum(-0.5 * np.log(2 * np.pi) - logs_p, axis=-1)[:, None]
        neg_cent2 = tf.matmul(-0.5 * (z_p ** 2), tf.transpose(s_p_sq_r, [0, 2, 1]))
        neg_cent3 = tf.matmul(z_p, tf.transpose((m_p * s_p_sq_r), [0, 2, 1]))
        neg_cent4 = tf.reduce_sum(-0.5 * (m_p ** 2) * s_p_sq_r, axis=-1)[:, None]
        neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

        attnmask = y_mask[..., None] * x_mask[:, None]
        attnmask = attnmask[:, :, :, 0]
        attn = tf.stop_gradient(self.monotonic_alignment_search(neg_cent, attnmask))

        m_p = tf.matmul(attn, m_p)
        logs_p = tf.matmul(attn, logs_p)

        w = tf.reduce_sum(attn, 1)

        if self.use_sdp:
            l_length = self.dp(tf.stop_gradient(x), x_mask, tf.expand_dims(w, -1), g=g, training=training)
            l_length = l_length / tf.reduce_sum(x_mask)
            l_length = tf.reduce_sum(l_length)
        else:
            logw = self.dp(tf.stop_gradient(x), x_mask, g=g, training=training)
            logw = tf.squeeze(logw, axis=-1)

            loss_f = tf.losses.mean_squared_error
            log_duration = tf.math.log(tf.cast(tf.math.add(w, 1), tf.float32)) * x_mask[:, :, 0]
            l_length = loss_f(log_duration, logw)

            # l_length = tf.reduce_sum(tf.square(logw - logw_), axis=[1, 2]) / tf.cast(y_lengths, tf.float32)
            # l_length = tf.reduce_mean(l_length)

            # l_length = tf.reduce_sum((logw - logw_)**2, [1, 2]) / tf.reduce_sum(x_mask)
            # l_length = tf.reduce_sum(l_length)

        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)
        return o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, x, x_lengths, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1.,
              training=False):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, training=training)
        if self.n_speakers > 0:
            g = tf.expand_dims(self.emb_g(sid), -1)
        else:
            g = None

        if self.use_sdp:
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w, training=training)
            w = tf.math.exp(logw) * x_mask
            w_ceil = tf.math.ceil(w * length_scale)[:, :, 0]
        else:
            logw = self.dp(x, x_mask, g=g, training=training)
            w = tf.nn.relu(tf.math.exp(logw) - 1.0) * x_mask
            w_ceil = tf.math.round(w * length_scale)[:, :, 0]

        # y_mask = tf.expand_dims(commons.sequence_mask(tf.reduce_sum(y_lengths, [1, 2]), None), 2)
        attn, y_mask = self.align(w_ceil)
        y_mask = tf.expand_dims(y_mask, 2)

        m_p = tf.matmul(attn, m_p)
        logs_p = tf.matmul(attn, logs_p)

        z_p = m_p + tf.random.normal(tf.shape(m_p)) * tf.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask), g=g)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)

    def mask(self, lengths: tf.Tensor, maxlen: Optional[tf.Tensor] = None):
        """Generate sequence mask from lengths.
        Args:
            lengths: [tf.int32; [B]], lengths.
            maxlen: tf.float32, maximum length.
        Returns:
            [tf.flaot32; [B, maxlen]], sequence mask.
        """
        if maxlen is None:
            maxlen = tf.reduce_max(lengths)
        # [B, S]
        return tf.cast(tf.range(maxlen)[None] < lengths[:, None], tf.float32)

    def align(self, duration: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Generate attention from duration.
        Args:
            duration: [tf.int32; [B, S]], duration vectors.
        Returns:
            attn: [tf.float32; [B, T, S]], attention map.
            mask: [tf.float32; [B, T]], sequence mask.
                where T = max(duration.sum(axis=-1)).
        """
        # S
        duration = tf.cast(duration, tf.int32)
        bsize, inplen = shape_list(duration)
        # T
        maxlen = tf.reduce_max(tf.reduce_sum(duration, axis=-1))
        # [B x S]
        cumsum = tf.reshape(tf.math.cumsum(duration, axis=-1), [-1])
        # [B x S, T]
        cumattn = self.mask(cumsum, maxlen)
        # [B, S, T]
        cumattn = tf.reshape(cumattn, [bsize, inplen, maxlen])
        # [B, S, T]
        attn = cumattn - tf.pad(cumattn, [[0, 0], [1, 0], [0, 0]])[:, : -1]
        # [B, T]
        mask = cumattn[:, -1]
        # [B, T, S], [B, T]
        return tf.transpose(attn, [0, 2, 1]), mask

    def monotonic_alignment_search(self,
                                   ll: tf.Tensor,
                                   mask: tf.Tensor) -> tf.Tensor:
        """
        Monotonic aligment search, reference from jaywalnut310's glow-tts.
        https://github.com/jaywalnut310/glow-tts/blob/master/commons.py#L60
        Args:
            ll: [tf.float32; [B, T, S]], loglikelihood matrix.
            mask: [tf.float32; [B, T, S]], attention mask.
        Returns:
            [tf.float32; [B, T, S]], alignment.
        """
        # B, T, S
        bsize, timestep, seqlen = shape_list(ll)
        # (expected) T x [B, S]
        direction = tf.TensorArray(dtype=tf.bool, size=timestep)
        # [B, S]
        prob = tf.zeros([bsize, seqlen], dtype=tf.float32)
        # [1, S]
        x_range = tf.expand_dims(tf.range(seqlen), 0)

        def condition(j, direction, prob):
            return j < timestep

        def body(j, direction, prob):
            prev = tf.pad(prob, [[0, 0], [1, 0]],
                          mode='CONSTANT',
                          constant_values=tf.float32.min)[:, :-1]
            cur = prob
            # larger value mask
            max_mask = tf.math.greater_equal(cur, prev)
            # select larger value
            prob_max = tf.where(max_mask, cur, prev)
            # write direction
            direction = direction.write(j, max_mask)
            # update prob

            x_range_ = tf.tile(x_range, [tf.shape(prob_max)[0], 1])
            j_ = tf.fill(tf.shape(x_range), j)
            min_ = tf.fill(tf.shape(x_range_), tf.float32.min)
            prob = tf.where(tf.math.less_equal(x_range_, j_),
                            prob_max + ll[:, j], min_)

            return j + 1, direction, prob

        init_state = (0, direction, prob)
        j, direction, prob = tf.while_loop(condition, body, init_state)
        # return direction.stack()
        direction = tf.cast(tf.transpose(direction.stack(), [1, 0, 2]), tf.int32)
        direction.set_shape((None, None, None))

        correct = tf.fill(tf.shape(direction), 1)
        direction = tf.where(tf.cast(mask, tf.bool), direction, correct)
        # (expected) T x [B, S]
        attn = tf.TensorArray(dtype=tf.float32, size=timestep)
        # [B]
        index = tf.cast(tf.reduce_sum(mask[:, 0], axis=-1), tf.int32) - 1
        # [B], [B]
        index_range, values = tf.range(bsize), tf.ones(bsize)

        def condition(j, attn, index):
            return j >= 0

        def body(j, attn, index):

            attn = attn.write(j, tf.scatter_nd(
                tf.stack([index_range, index], axis=1),
                values, [bsize, seqlen]))
            # [B]
            dir = tf.gather_nd(
                direction,
                tf.stack([index_range, tf.cast(values, tf.int32) * j, index],
                         axis=1))
            # [B]
            index = index + dir - 1
            return j - 1, attn, index

        init_state = (timestep - 1, attn, index)
        _, attn, _ = tf.while_loop(condition, body, init_state)
        stacked = attn.stack()
        stacked = tf.transpose(stacked, [1, 0, 2])
        stacked.set_shape((None, None, None))
        return stacked * mask
