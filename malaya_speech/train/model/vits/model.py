"""
MIT License

Copyright (c) 2021 YoungJoong Kim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import Dict, Optional, Tuple

import tensorflow as tf
import numpy as np

from .config import Config
from .durator import DurationPredictor
from .flow import WaveNetFlow
from .posterior import PosteriorEncoder
from .transformer import Transformer
from .slicing import rand_slice_segments
from ..utils import shape_list


class Model(tf.keras.Model):
    DURATION_MAX = 80

    def __init__(self, config: Config):
        """Initializer.
        Args:
            config: configuration object.
        """
        super().__init__()
        self.mel = config.mel
        self.temperature = config.temperature
        self.length_scale = config.length_scale
        self.embedding = tf.keras.layers.Embedding(
            config.vocabs, config.channels)
        self.encoder = Transformer(config)
        self.enc_q = PosteriorEncoder(config)
        self.flow = WaveNetFlow(config)
        self.durator = DurationPredictor(
            config.dur_layers, config.channels,
            config.dur_kernel, config.dur_dropout)
        # mean-only training
        self.proj = tf.keras.layers.Dense(config.mel)
        self.interchannels = config.mel
        self.segment_size = config.segment_size
        self.hop_size = config.hop_size

    def call(self, inputs: tf.Tensor, lengths: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        """Generate mel-spectrogram from text inputs.
        Args:
            inputs: [tf.int32; [B, S]], input sequence.
            lengths: [tf.int32; [B]], input lengths.
        Returns:
            mel: [tf.float32; [B, T, M]], generated mel-spectrogram.
            mellen: [tf.int32; [B]], length of the mel-spectrogram.
            attn: [tf.float32; [B, T / F, S]], attention alignment.
        """
        # S
        _, seqlen = shape_list(inputs)
        mask = self.mask(lengths, maxlen=seqlen)
        # [B, S, C]
        embedding = self.embedding(inputs) * mask[..., None]
        # [B, S, C]
        hidden, _ = self.encoder(embedding, mask)
        # [B, S, 2C]
        mean = self.proj(hidden)
        # [B, S, C]
        duration = self.quantize(self.durator(hidden, mask), mask)
        # [B, T / F, S], [B, T / F]
        attn, mask = self.align(duration)

        # [B, T / F, N(=M x F)], []
        mean = tf.matmul(attn, mean)
        std = self.temperature
        # [B, T / F, M x F]
        sample = mean + tf.random.normal(tf.shape(mean)) * std

        # [B, T / F, M x F]
        mel = self.flow.inverse(sample * mask[..., None], mask)
        # [B]
        mellen = tf.cast(tf.reduce_sum(mask, axis=-1), tf.int32)
        return mel, mellen, attn

    def compute_loss(self, text: tf.Tensor, textlen: tf.Tensor,
                     mel: tf.Tensor, mellen: tf.Tensor) \
            -> Tuple[tf.Tensor, Dict[str, tf.Tensor], tf.Tensor]:
        """Compute loss for glow-tts.
        Args:
            text: [tf.int32; [B, S]], input text.
            textlen: [tf.int32; [B]], text lengths.
            mel: [tf.float32; [B, T, M]], mel-spectrogram.
            mellen: [tf.int32; [B]], mel lengths.
        Returns:
            loss: [tf.float32; []], loss tensor.
            ll: loss lists.
            attn: [tf.float32; [B, T, S]], attention alignment.
        """
        # [B, S]
        mask = self.mask(textlen, maxlen=tf.shape(text)[1])
        # [B, S, C]
        embedding = self.embedding(text) * mask[..., None]
        # [B, S, C]
        hidden, _ = self.encoder(embedding, mask)
        # [B, S, 2C]
        mean = self.proj(hidden)

        # [B, T]
        melmask = self.mask(mellen, maxlen=tf.shape(mel)[1])

        latent, mean_q = self.enc_q(mel, melmask)

        latent_p = self.flow(latent, melmask)

        # [B, T', S]
        attnmask = melmask[..., None] * mask[:, None]

        # s_p_sq_r = tf.exp(-2 * logs_p)
        # neg_cent1 = tf.reduce_sum(-0.5 * np.log(2 * np.pi) - logs_p, axis=-1)[:, None]
        # neg_cent2 = tf.matmul(-0.5 * (latent_p ** 2), tf.transpose(s_p_sq_r, [0, 2, 1]))
        # neg_cent3 = tf.matmul(latent_p, tf.transpose((mean * s_p_sq_r), [0, 2, 1]))
        # neg_cent4 = tf.reduce_sum(-0.5 * (mean ** 2) * s_p_sq_r, axis=-1)[:, None]
        # neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
        # attn = tf.stop_gradient(self.monotonic_alignment_search(neg_cent, attnmask))

        # x_s_sq_r = torch.exp(-2 * x_logs)
        # logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs, [1]).unsqueeze(-1) # [b, t, 1]
        # logp2 = torch.matmul(x_s_sq_r.transpose(1,2), -0.5 * (z ** 2)) # [b, t, d] x [b, d, t'] = [b, t, t']
        # logp3 = torch.matmul((x_m * x_s_sq_r).transpose(1,2), z) # [b, t, d] x [b, d, t'] = [b, t, t']
        # logp4 = torch.sum(-0.5 * (x_m ** 2) * x_s_sq_r, [1]).unsqueeze(-1) # [b, t, 1]
        # logp = logp1 + logp2 + logp3 + logp4 # [b, t, t']

        # (latent)** 2 - 2*(latent.mean) + (mean)**2

        dist = tf.reduce_sum(tf.square(latent_p), axis=-1, keepdims=True) - \
            2 * tf.matmul(latent_p, tf.transpose(mean, [0, 2, 1])) + \
            tf.reduce_sum(tf.square(mean), axis=-1)[:, None]
        # [B, T', S], assume constant standard deviation, 1.
        ll = -0.5 * (np.log(2 * np.pi) + dist) * attnmask
        # [B, T', S], attention alignment
        attn = tf.stop_gradient(self.monotonic_alignment_search(ll, attnmask))

        mean = tf.matmul(attn, mean)

        nll = 0.5 * (np.log(2 * np.pi) +
                     tf.reduce_sum(tf.square(latent_p - mean), axis=[1, 2]))
        # []
        nll = tf.reduce_mean(
            nll / tf.cast(mellen * tf.shape(mean)[-1], tf.float32))

        # [B, S]
        logdur = self.durator(tf.stop_gradient(hidden), mask)
        # [B, S]
        gtdur = tf.math.log(tf.maximum(tf.reduce_sum(attn, axis=1), 1e-5)) * mask
        # [B]
        durloss = tf.reduce_sum(tf.square(logdur - gtdur), axis=-1) / \
            tf.cast(mellen, tf.float32)
        # []
        durloss = tf.reduce_mean(durloss)

        z_slice, ids_slice = rand_slice_segments(latent, mellen, self.segment_size // self.hop_size, np.log(1e-2))

        return {'nll': nll, 'durloss': durloss}, attn, latent, z_slice, ids_slice

    def quantize(self, logdur: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """Convert log-duration to duration.
        Args:
            logdur: [tf.float32; [B, T]], log-duration.
            mask: [tf.float32; [B, T]], sequence mask.
        Returns:
            [tf.int32; [B, T]], duration.
        """
        # [B, T]
        dur = tf.round(tf.exp(logdur))
        # [B, T]
        dur = tf.clip_by_value(dur, 1., Model.DURATION_MAX) * mask * self.length_scale
        return tf.cast(dur, tf.int32)

    def mask(self, lengths: tf.Tensor, maxlen: Optional[tf.Tensor] = None) \
            -> tf.Tensor:
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
        attn = cumattn - tf.pad(cumattn, [[0, 0], [1, 0], [0, 0]])[:, :-1]
        # [B, T]
        mask = cumattn[:, -1]
        # [B, T, S], [B, T]
        return tf.transpose(attn, [0, 2, 1]), mask

    def monotonic_alignment_search(self,
                                   ll: tf.Tensor,
                                   mask: tf.Tensor) -> tf.Tensor:
        """Monotonic aligment search, reference from jaywalnut310's glow-tts.
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
