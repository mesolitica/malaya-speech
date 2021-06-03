import tensorflow as tf
from dataclasses import dataclass, field
from .layer import gelu, glu, GumbelVectorQuantizer
from .masking import index_put, index_put_constant
from ..utils import shape_list


class Model(tf.keras.Model):
    def __init__(self, cfg, **kwargs):
        super(Model, self).__init__(name='wav2vec2', **kwargs)
        self.cfg = cfg

        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]

        self.quantizer = None

        self.n_negatives = cfg.num_negatives
        self.cross_sample_negatives = cfg.cross_sample_negatives
        self.codebook_negatives = cfg.codebook_negatives
        self.negatives_from_everywhere = cfg.negatives_from_everywhere

        self.logit_temp = cfg.logit_temp

        final_dim = cfg.final_dim if cfg.final_dim > 0 else self.embed

        if cfg.quantize_targets:
            vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else final_dim
            self.quantizer = GumbelVectorQuantizer(
                dim=self.embed,
                num_vars=cfg.latent_vars,
                temp=cfg.latent_temp,
                groups=cfg.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
            )
        self.project_q = tf.keras.layers.Dense(final_dim)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = tf.keras.layers.Dense(
                final_dim * 2, activation=glu
            )

        self.final_proj = tf.keras.layers.Dense(final_dim)

    def sample_negatives(self, y, num, padding_count=None):
        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return tf.zeros(shape=(0,), dtype=tf.int32)
        bsz, tsz, fsz = shape_list(y)
        y = tf.reshape(y, (-1, fsz))
        cross_high = tsz * bsz
        high = tsz - (padding_count or 0)

        if self.n_negatives > 0:
            tszs = tf.reshape(
                tf.tile(
                    tf.expand_dims(tf.range(num), -1), (1, self.n_negatives)
                ),
                [-1],
            )
            neg_idxs = tf.random.uniform(
                minval=0,
                maxval=high - 1,
                shape=(bsz, self.n_negatives * num),
                dtype=tf.int32,
            )

            mask = neg_idxs >= tszs
            neg_idxs = tf.where(mask, neg_idxs + 1, neg_idxs)
        if self.cross_sample_negatives > 0:
            tszs = tf.reshape(
                tf.tile(
                    tf.expand_dims(tf.range(num), -1),
                    (1, self.cross_sample_negatives),
                ),
                [-1],
            )
            cross_neg_idxs = tf.random.uniform(
                minval=0,
                maxval=cross_high - 1,
                shape=(bsz, self.cross_sample_negatives * num),
                dtype=tf.int32,
            )
            mask = cross_neg_idxs >= tszs
            cross_neg_idxs = tf.where(mask, cross_neg_idxs + 1, cross_neg_idxs)

        if self.n_negatives > 0:
            i = tf.range(tf.shape(neg_idxs)[0])
            i = tf.tile(tf.expand_dims(i, -1), (1, tf.shape(neg_idxs)[1]))
            neg_idxs += i * high
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = tf.concat([neg_idxs, cross_neg_idxs], axis=1)

        negs = tf.gather(y, tf.reshape(neg_idxs, [-1]))
        negs = tf.reshape(
            negs,
            (bsz, num, self.n_negatives + self.cross_sample_negatives, fsz),
        )
        negs = tf.transpose(negs, (2, 0, 1, 3))
        return negs, neg_idxs

    def compute_preds(self, x, y, negatives):
        tiled = tf.tile(tf.expand_dims(y, 0), (tf.shape(negatives)[0], 1, 1, 1))
        neg_is_pos = tf.reduce_all(tf.equal(tiled, negatives), axis=-1)
        y = tf.expand_dims(y, 0)
        targets = tf.concat([y, negatives], axis=0)
        x = tf.tile(tf.expand_dims(x, 0), (tf.shape(targets)[0], 1, 1, 1))
        logits = -(
            tf.losses.cosine_distance(
                tf.nn.l2_normalize(x, -1),
                tf.nn.l2_normalize(targets, -1),
                dim=-1,
                reduction='none',
            )
            - 1
        )
        logits = logits[:, :, :, 0]
        logits = logits / self.logit_temp
        left, right = logits[:1], logits[1:]
        right = tf.cond(
            tf.reduce_any(neg_is_pos),
            lambda: index_put_constant(right, neg_is_pos, -1e9),
            lambda: right,
        )
        logits = tf.concat([left, right], axis=0)
        return logits

    def call(self, x, y, training=True):

        if self.quantizer:
            q = self.quantizer(y, produce_targets=False)
            y = q['x']
            num_vars = q['num_vars']
            code_ppl = q['code_perplexity']
            prob_ppl = q['prob_perplexity']
            curr_temp = q['temp']

            y = self.project_q(y)

            negs, _ = self.sample_negatives(y, tf.shape(y)[1])

        else:
            y = self.project_q(y)
            negs, _ = self.sample_negatives(y, tf.shape(y)[1])

        if self.target_glu:
            y = self.target_glu(y)
            negs = self.target_glu(negs)

        x = self.final_proj(x)
        x = self.compute_preds(x, y, negs)
        result = {'x': x}

        if prob_ppl is not None:
            result['prob_perplexity'] = prob_ppl
            result['code_perplexity'] = code_ppl

        return result, float(num_vars), curr_temp
