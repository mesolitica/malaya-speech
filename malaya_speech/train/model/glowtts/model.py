import math
import tensorflow as tf
from . import alignment, attention, common, modules
from ..utils import shape_list
from ..fastspeech.model import TFFastSpeechEmbeddings, TFFastSpeechEncoder, TFTacotronPostnet
from ..fastspeech2.model import FastSpeechVariantPredictor
import logging


class Encoder(tf.keras.layers.Layer):
    def __init__(self, config, config_glowtts, **kwargs):
        super(Encoder, self).__init__(name='Encoder', **kwargs)
        self.embeddings = TFFastSpeechEmbeddings(config, name='embeddings')
        self.encoder = TFFastSpeechEncoder(
            config.encoder_self_attention_params, name='encoder'
        )
        self.proj_w = FastSpeechVariantPredictor(
            config, dtype=tf.float32, name='duration_predictor'
        )
        hidden_size = config.encoder_self_attention_params.hidden_size
        self.prenet = config_glowtts.prenet
        if self.prenet:
            self.pre = modules.ConvReluNorm(hidden_size, hidden_size,
                                            kernel_size=5, n_layers=3, p_dropout=0.5)

        self.proj_m = tf.keras.layers.Conv1D(config.num_mels, 1, padding='same')
        self.mean_only = config_glowtts.mean_only
        if not self.mean_only:
            self.proj_s = tf.keras.layers.Conv1D(config.num_mels, 1, padding='same')

    def call(
        self,
        input_ids,
        training=True,
        **kwargs,
    ):
        speaker_ids = tf.convert_to_tensor([0], tf.int32)
        attention_mask = tf.math.not_equal(input_ids, 0)
        embedding_output = self.embeddings(
            [input_ids, speaker_ids], training=training
        )
        attention_mask = tf.cast(attention_mask, tf.float32)
        attention_mask_expand = tf.expand_dims(attention_mask, -1)
        if self.prenet:
            embedding_output = self.pre(embedding_output, attention_mask_expand)
        x = self.encoder(
            [embedding_output, attention_mask], training=training
        )
        x = x[0]
        x_dp = tf.identity(x)
        x_m = self.proj_m(x) * attention_mask_expand
        if not self.mean_only:
            x_logs = self.proj_s(x) * attention_mask_expand
        else:
            x_logs = tf.zeros_like(x_m)
        logw = self.proj_w(
            [x_dp, speaker_ids, attention_mask]
        )
        logw = tf.expand_dims(logw, -1)
        return x_m, x_logs, logw, attention_mask_expand


class FlowSpecDecoder(tf.keras.layers.Layer):
    def __init__(self, config, config_glowtts, **kwargs):
        super(FlowSpecDecoder, self).__init__(name='FlowSpecDecoder', **kwargs)
        self.in_channels = config.num_mels
        self.hidden_channels = config_glowtts.hidden_channels
        self.kernel_size = config_glowtts.kernel_size
        self.n_layers = config_glowtts.n_block_layers
        self.dilation_rate = config_glowtts.dilation_rate
        self.n_blocks = config_glowtts.n_blocks_dec
        self.p_dropout = config_glowtts.p_dropout_dec
        self.n_split = config_glowtts.n_split
        self.n_sqz = config_glowtts.n_sqz
        self.sigmoid_scale = config_glowtts.sigmoid_scale
        self.gin_channels = config_glowtts.gin_channels

        self.flows = []
        for b in range(self.n_blocks):
            with tf.variable_scope(f'n_blocks_{b + 1}') as vs:
                self.flows.append(modules.ActNorm(channels=self.in_channels *
                                  self.n_sqz, ddi=config_glowtts.ddi, name=b,))
                self.flows.append(modules.InvConvNear(channels=self.in_channels *
                                  self.n_sqz, n_split=self.n_split, name=b))
                self.flows.append(attention.CouplingBlock(
                    self.in_channels * self.n_sqz,
                    self.hidden_channels,
                    kernel_size=self.kernel_size,
                    dilation_rate=self.dilation_rate,
                    n_layers=self.n_layers,
                    gin_channels=self.gin_channels,
                    p_dropout=self.p_dropout,
                    sigmoid_scale=self.sigmoid_scale, name=b))

    def store_inverse(self):
        for f in self.flows:
            f.store_inverse()

    def call(self, x, x_mask, g=None, reverse=False, training=True, **kwargs):
        if not reverse:
            flows = self.flows
            logdet_tot = 0
        else:
            flows = reversed(self.flows)
            logdet_tot = None

        if self.n_sqz > 1:
            x, x_mask = common.squeeze(x, x_mask, self.n_sqz)

        for f in flows:
            if not reverse:
                x, logdet = f(x, x_mask, g=g, reverse=reverse, training=training)
                logdet_tot += logdet
            else:
                x, logdet = f(x, x_mask, g=g, reverse=reverse, training=training)

            logging.debug(f'{f} {x} {logdet}')

        if self.n_sqz > 1:
            x, x_mask = common.unsqueeze(x, x_mask, self.n_sqz)

        return x, logdet_tot


class Model(tf.keras.Model):
    def __init__(self, config, config_glowtts, **kwargs):
        super(Model, self).__init__(name='glowtts', **kwargs)
        self.encoder = Encoder(config, config_glowtts)
        self.decoder = FlowSpecDecoder(config, config_glowtts)
        self.config = config
        self.config_glowtts = config_glowtts

    def preprocess(self, y, y_lengths, y_max_length):
        if y_max_length is not None:
            y_max_length = (y_max_length // self.config_glowtts.n_sqz) * self.config_glowtts.n_sqz
            y = y[:, :y_max_length]
        y_lengths = (y_lengths // self.config_glowtts.n_sqz) * self.config_glowtts.n_sqz
        return y, y_lengths, y_max_length

    def call(self, x, y=None, y_lengths=None, training=True, gen=False,
             noise_scale=1., length_scale=1., **kwargs):

        # [B, tx, C], [B, tx, C], [B, tx, 1], [B, tx, 1]
        x_m, x_logs, logw, x_mask = self.encoder(x, training=training)
        if gen:
            w = tf.math.exp(logw) * x_mask * length_scale
            w_ceil = tf.math.ceil(w)
            y_lengths = tf.reduce_sum(w_ceil, [2, 1])
            y_lengths = tf.clip_by_value(y_lengths, 1, tf.reduce_max(y_lengths))
            y_lengths = tf.cast(y_lengths, tf.float32)
            y_max_length = None
        else:
            y_max_length = tf.shape(y)[1]

        y, y_lengths, y_max_length = self.preprocess(y, y_lengths, y_max_length)

        # [B, ty]
        z_mask = tf.sequence_mask(y_lengths, y_max_length)
        # [B, 1, ty]
        z_mask = tf.expand_dims(z_mask, 1)
        # [B, 1, ty]
        z_mask = tf.cast(z_mask, x_mask.dtype)

        # [B, 1, tx, 1] * [B, 1, 1, ty] = [B, 1, tx, ty]
        attn_mask = tf.expand_dims(tf.transpose(x_mask, [0, 2, 1]), -1) * tf.expand_dims(z_mask, 2)
        # [B, ty, 1]
        z_mask = tf.transpose(z_mask, [0, 2, 1])

        if gen:
            # [B, tx, ty]
            attn = common.generate_path(tf.squeeze(w_ceil, 2), tf.squeeze(attn_mask, 1))
            # [B, 1, tx, ty]
            attn = tf.expand_dims(attn, 1)
            # [B, ty, tx]
            left = tf.transpose(tf.squeeze(attn, 1), [0, 2, 1])
            # [B, ty, tx] x [B, tx, C] -> [B, ty, C]
            z_m = tf.matmul(tf.transpose(tf.squeeze(attn, 1), [0, 2, 1]), x_m)
            # [B, ty, tx] x [B, tx, C] -> [B, ty, C]
            z_logs = tf.matmul(tf.transpose(tf.squeeze(attn, 1), [0, 2, 1]), x_logs)

            # [B, tx, 1] * [B, tx, 1]
            logw_ = tf.math.log(1e-8 + tf.transpose(tf.reduce_sum(attn, -1), [0, 2, 1])) * x_mask

            z = (z_m + tf.math.exp(z_logs) * tf.random.normal(shape=shape_list(z_m)) * noise_scale) * z_mask
            y, logdet = self.decoder(z, z_mask, reverse=True, training=training)
            return (y, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_)
        else:
            # [B, ty, C]
            z, logdet = self.decoder(y, z_mask, reverse=False, training=training)
            # [B, C, ty]
            z = tf.transpose(z, [0, 2, 1])
            # [B, 1, tx]
            x_mask = tf.transpose(x_mask, [0, 2, 1])
            # [B, C, tx]
            x_logs = tf.transpose(x_logs, [0, 2, 1])
            # [B, C, tx]
            x_m = tf.transpose(x_m, [0, 2, 1])
            # [B, C, tx]
            x_s_sq_r = tf.stop_gradient(tf.math.exp(-2 * x_logs))
            # [B, tx, 1]
            logp1 = tf.stop_gradient(tf.expand_dims(tf.reduce_sum(-0.5 * math.log(2 * math.pi) - x_logs, 1), -1))
            # [B, tx, 1] x [B, C, ty] = [B, tx, ty]
            logp2 = tf.stop_gradient(tf.matmul(tf.transpose(x_s_sq_r, [0, 2, 1]), -0.5 * (z ** 2)))
            # [B, tx, C] x [B, C, ty] = [B, tx, ty]
            logp3 = tf.stop_gradient(tf.matmul(tf.transpose(x_m * x_s_sq_r, [0, 2, 1]), z))
            # [B, tx, 1]
            logp4 = tf.stop_gradient(tf.expand_dims(tf.reduce_sum(-0.5 * (x_m ** 2) * x_s_sq_r, 1), -1))
            # [B, tx, 1] + [B, tx, ty] + [B, tx, ty] + [B, tx, 1] = [B, tx, ty]
            logp = tf.stop_gradient(logp1 + logp2 + logp3 + logp4)

            # attn = tf.compat.v1.numpy_function(common.maximum_path,
            #                                    [logp, tf.squeeze(attn_mask, 1)], tf.float32)
            print(logp, attn_mask)
            # [B, tx, ty], [B, tx, ty]
            attn = tf.compat.v1.numpy_function(alignment.maximum_path,
                                               [logp, tf.squeeze(attn_mask, 1)], tf.float32)
            # [B, tx, ty]
            attn.set_shape((None, None, None))
            # [B, 1, tx, ty]
            attn = tf.expand_dims(attn, 1)
            # [B, ty, tx] x [B, tx, C] = [B, ty, C]
            z_m = tf.matmul(tf.transpose(tf.squeeze(attn, 1), [0, 2, 1]), tf.transpose(x_m, [0, 2, 1]))
            # [B, C, ty]
            z_m = tf.transpose(z_m, [0, 2, 1])

            # [B, ty, tx] x [B, tx, C] = [B, ty, C]
            z_logs = tf.matmul(tf.transpose(tf.squeeze(attn, 1), [0, 2, 1]), tf.transpose(x_logs, [0, 2, 1]))
            # [B, C, ty]
            z_logs = tf.transpose(z_logs, [0, 2, 1])

            # [B, 1, tx] x [B, 1, tx]
            logw_ = tf.math.log(1e-8 + tf.reduce_sum(attn, -1)) * x_mask

            return (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_)
