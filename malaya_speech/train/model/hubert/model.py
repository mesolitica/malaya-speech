import tensorflow as tf
from dataclasses import dataclass, field
from .layer import gelu, glu, ConvFeatureExtractionModel
from .masking import compute_mask_indices, index_put, index_put_constant
from ..utils import shape_list
import numpy as np


class Model(tf.keras.Model):
    def __init__(self, cfg, encoder, dictionary=None, **kwargs):
        super(Model, self).__init__(name='hubert', **kwargs)
        self.cfg = cfg

        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )
        feature_ds_rate = np.prod([s for _, _, s in feature_enc_layers])
        self.feat2tar_ratio = (
            cfg.label_rate * feature_ds_rate / cfg.sample_rate
        )

        self.post_extract_proj = (
            tf.keras.layers.Dense(cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = tf.keras.layers.Dropout(cfg.dropout_input)
        self.dropout_features = tf.keras.layers.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult
        self.logit_temp = cfg.logit_temp
        self.skip_masked = cfg.skip_masked
        self.skip_nomask = cfg.skip_nomask

        final_dim = (
            cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim
        )

        self.mask_emb = tf.get_variable(
            name='mask_emb',
            shape=[cfg.encoder_embed_dim],
            initializer=tf.truncated_normal_initializer(),
        )
        self.encoder = encoder
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = tf.keras.layers.Dense(
                final_dim * 2, activation=glu
            )
        self.untie_final_proj = cfg.untie_final_proj
        if self.untie_final_proj:
            self.final_proj = tf.keras.layers.Dense(final_dim * len(dictionaries))
        else:
            self.final_proj = tf.keras.layers.Dense(final_dim)

        if dictionary is not None:
            self.num_classes = len(dictionary)
            self.label_embs_concat = tf.get_variable(
                name='label_embs_concat',
                shape=[self.num_classes, final_dim],
                initializer=tf.truncated_normal_initializer(),
            )
        else:
            print('cannot find dictionary. assume will be used for fine-tuning')

    def _get_feat_extract_output_lengths(self, input_lengths):
        def _conv_out_length(input_length, kernel_size, stride):
            return tf.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = eval(self.cfg.conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(
                input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2]
            )

        return tf.cast(input_lengths, tf.int32)

    def apply_mask(
        self, x, padding_mask, mask_indices=None, mask_channel_indices=None
    ):
        B, T, C = shape_list(x)
        if self.mask_prob > 0:
            mask_indices = tf.compat.v1.numpy_function(
                compute_mask_indices,
                [
                    (B, T),
                    padding_mask,
                    self.mask_prob,
                    self.mask_length,
                    self.mask_selection,
                    self.mask_other,
                    2,
                    self.no_mask_overlap,
                    self.mask_min_space,
                ],
                tf.bool,
            )
            mask_indices.set_shape((None, None))
            self.mask_indices = mask_indices
            x = index_put(x, mask_indices, self.mask_emb)
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = tf.compat.v1.numpy_function(
                compute_mask_indices,
                [
                    (B, C),
                    'None',
                    self.mask_channel_prob,
                    self.mask_channel_length,
                    self.mask_channel_selection,
                    self.mask_channel_other,
                    0,
                    self.no_mask_channel_overlap,
                    self.mask_channel_min_space,
                ],
                tf.bool,
            )
            mask_channel_indices.set_shape((None, None))
            mask_channel_indices = tf.expand_dims(mask_channel_indices, 1)
            mask_channel_indices = tf.tile(mask_channel_indices, (1, T, 1))
            self.mask_channel_indices = mask_channel_indices
            x = index_put_constant(x, mask_channel_indices, 0.0)
        return x, mask_indices

    def compute_nce(self, x, pos, negs):
        tiled = tf.tile(tf.expand_dims(pos, 0), (tf.shape(negs)[0], 1, 1))
        neg_is_pos = tf.reduce_all(tf.equal(tiled, negs), axis=-1)
        pos = tf.expand_dims(pos, 0)
        targets = tf.concat([pos, negs], axis=0)
        x = tf.tile(tf.expand_dims(x, 0), (tf.shape(targets)[0], 1, 1,))
        logits = -(
            tf.losses.cosine_distance(
                tf.nn.l2_normalize(x, -1),
                tf.nn.l2_normalize(targets, -1),
                dim=-1,
                reduction='none',
            )
            - 1
        )
        logits = logits[:, :, 0]
        logits = logits / self.logit_temp
        left, right = logits[:1], logits[1:]
        right = tf.cond(
            tf.reduce_any(neg_is_pos),
            lambda: index_put_constant(right, neg_is_pos, -1e9),
            lambda: right,
        )
        logits = tf.concat([left, right], axis=0)
        return tf.transpose(logits, [1, 0])

    def forward_targets(self, features, target_list):
        feat_tsz = tf.shape(features)[1]
        targ_tsz = tf.shape(target_list)[1]

        def f1():
            feat_tsz = tf.cast(targ_tsz / self.feat2tar_ratio, tf.int32)
            return features[:, :feat_tsz], feat_tsz

        features, feat_tsz = tf.cond(
            self.feat2tar_ratio * feat_tsz > targ_tsz, f1, lambda: (features, feat_tsz)
        )
        target_inds = tf.cast(tf.range(feat_tsz), tf.float32) * self.feat2tar_ratio
        target_inds = tf.cast(target_inds, tf.int32)
        target_list = tf.gather(target_list, target_inds, axis=1)
        return features, target_list

    def call(
        self,
        source,
        target_list=None,
        padding_mask=None,
        mask=True,
        features_only=False,
        output_layer=None,
        training=True,
    ):
        features = self.feature_extractor(source, training=training)
        if self.feature_grad_mult <= 0:
            features = tf.stop_gradient(features)

        # if target_list is not None:
        #     features, target_list = self.forward_targets(features, target_list)

        features_pen = tf.reduce_mean(tf.math.pow(features, 2))
        features = self.layer_norm(features, training=training)
        unmasked_features = tf.identity(features, name='unmasked_features')
        if padding_mask is not None:
            input_lengths = padding_mask
            output_lengths = self._get_feat_extract_output_lengths(
                input_lengths
            )

            max_length = tf.cast(tf.reduce_max(output_lengths), tf.int32)
            padding_mask = tf.sequence_mask(
                lengths=output_lengths,
                maxlen=max_length,
                dtype=tf.float32,
            )
            padding_mask.set_shape((None, None))
            padding_mask = tf.math.logical_not(tf.cast(padding_mask, tf.bool))
            self.padding_mask = padding_mask

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features, training=training)
        unmasked_features = self.dropout_features(
            unmasked_features, training=training
        )

        if mask:
            x, mask_indices = self.apply_mask(
                features, padding_mask, target_list
            )
        else:
            x = features
            mask_indices = None

        x = self.encoder(x, padding_mask, training=training)

        if features_only:
            return {'x': x, 'padding_mask': padding_mask}

        def compute_pred(proj_x, target, label_embs):
            y = tf.gather(label_embs, target, axis=0)
            negs = tf.expand_dims(label_embs, 1)
            negs = tf.tile(negs, [1, tf.shape(proj_x)[0], 1])
            if self.target_glu:
                y = self.target_glu(y)
                negs = self.target_glu(negs)
            return self.compute_nce(proj_x, y, negs)

        label_embs_list = self.label_embs_concat

        if not self.skip_masked:
            masked_indices = tf.math.logical_and(tf.math.logical_not(padding_mask), mask_indices)
            proj_x_m = self.final_proj(tf.boolean_mask(x, masked_indices))
            masked_target_list = tf.boolean_mask(target_list, masked_indices)
            logit_m_list = compute_pred(proj_x_m, masked_target_list, label_embs_list)

        else:
            logit_m_list = None

        if not self.skip_nomask:
            masked_indices = tf.math.logical_and(tf.math.logical_not(padding_mask),
                                                 tf.math.logical_not(mask_indices))
            proj_x_m = self.final_proj(tf.boolean_mask(x, masked_indices))
            masked_target_list = tf.boolean_mask(target_list, masked_indices)
            logit_u_list = compute_pred(proj_x_m, masked_target_list, label_embs_list)
        else:
            logit_u_list = None

        result = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
            'x': x,
        }
        return result
