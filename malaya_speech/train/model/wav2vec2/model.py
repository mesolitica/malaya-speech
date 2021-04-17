import tensorflow as tf
from dataclasses import dataclass, field
from .layer import gelu, glu, ConvFeatureExtractionModel, GumbelVectorQuantizer
from .masking import compute_mask_indices, index_put, index_put_constant
from ..utils import shape_list


class Model(tf.keras.Model):
    def __init__(self, cfg, encoder, **kwargs):
        super(Model, self).__init__(name = 'wav2vec2', **kwargs)
        self.cfg = cfg

        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers = feature_enc_layers,
            dropout = 0.0,
            mode = cfg.extractor_mode,
            conv_bias = cfg.conv_bias,
        )
        self.layer_norm = tf.keras.layers.LayerNormalization()

        self.post_extract_proj = (
            tf.keras.layers.Dense(cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim and not cfg.quantize_input
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

        self.quantizer = None
        self.input_quantizer = None

        self.n_negatives = cfg.num_negatives
        self.cross_sample_negatives = cfg.cross_sample_negatives
        self.codebook_negatives = cfg.codebook_negatives
        self.negatives_from_everywhere = cfg.negatives_from_everywhere

        self.logit_temp = cfg.logit_temp

        final_dim = (
            cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim
        )

        if cfg.quantize_targets:
            vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else final_dim
            self.quantizer = GumbelVectorQuantizer(
                dim = self.embed,
                num_vars = cfg.latent_vars,
                temp = cfg.latent_temp,
                groups = cfg.latent_groups,
                combine_groups = False,
                vq_dim = vq_dim,
            )
        self.project_q = tf.keras.layers.Dense(final_dim)

        if cfg.quantize_input:
            if cfg.same_quantizer and self.quantizer is not None:
                vq_dim = final_dim
                self.input_quantizer = self.quantizer
            else:
                vq_dim = (
                    cfg.latent_dim
                    if cfg.latent_dim > 0
                    else cfg.encoder_embed_dim
                )
                self.input_quantizer = GumbelVectorQuantizer(
                    dim = self.embed,
                    num_vars = cfg.latent_vars,
                    temp = cfg.latent_temp,
                    groups = cfg.latent_groups,
                    combine_groups = False,
                    vq_dim = vq_dim,
                )

        self.mask_emb = tf.get_variable(
            name = 'mask_emb',
            shape = [cfg.encoder_embed_dim],
            initializer = tf.truncated_normal_initializer(),
        )
        self.encoder = encoder
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = tf.keras.layers.Dense(
                final_dim * 2, activation = glu
            )

        self.final_proj = tf.keras.layers.Dense(final_dim)

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
        self, x, padding_mask, mask_indices = None, mask_channel_indices = None
    ):
        B, T, C = shape_list(x)
        if self.mask_prob > 0:
            if mask_indices is None:
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
            if mask_channel_indices is None:
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

    def sample_negatives(self, y, num, padding_count = None):
        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return tf.zeros(shape = (0,), dtype = tf.int32)
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
                minval = 0,
                maxval = high - 1,
                shape = (bsz, self.n_negatives * num),
                dtype = tf.int32,
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
                minval = 0,
                maxval = cross_high - 1,
                shape = (bsz, self.cross_sample_negatives * num),
                dtype = tf.int32,
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
            neg_idxs = tf.concat([neg_idxs, cross_neg_idxs], axis = 1)

        negs = tf.gather(y, tf.reshape(neg_idxs, [-1]))
        negs = tf.reshape(
            negs,
            (bsz, num, self.n_negatives + self.cross_sample_negatives, fsz),
        )
        negs = tf.transpose(negs, (2, 0, 1, 3))
        return negs, neg_idxs

    def compute_preds(self, x, y, negatives):
        y = tf.expand_dims(y, 0)
        targets = tf.concat([y, negatives], axis = 0)
        x = tf.tile(tf.expand_dims(x, 0), (tf.shape(targets)[0], 1, 1, 1))
        logits = -(
            tf.losses.cosine_distance(
                tf.nn.l2_normalize(x, -1),
                tf.nn.l2_normalize(targets, -1),
                dim = -1,
                reduction = 'none',
            )
            - 1
        )
        logits = logits[:, :, :, 0]
        logits = logits / self.logit_temp
        return logits

    def call(
        self,
        source,
        padding_mask = None,
        mask = True,
        features_only = False,
        mask_indices = None,
        mask_channel_indices = None,
        padding_count = None,
        training = True,
    ):
        features = self.feature_extractor(source, training = training)
        if self.feature_grad_mult <= 0:
            features = tf.stop_gradient(features)

        features_pen = tf.reduce_mean(tf.math.pow(features, 2))
        features = self.layer_norm(features, training = training)
        unmasked_features = features
        if padding_mask is not None:
            input_lengths = padding_mask
            output_lengths = self._get_feat_extract_output_lengths(
                input_lengths
            )

            max_length = tf.cast(tf.reduce_max(output_lengths), tf.int32)
            padding_mask = tf.sequence_mask(
                lengths = output_lengths,
                maxlen = max_length,
                dtype = tf.float32,
            )
            padding_mask.set_shape((None, None))
            padding_mask = tf.math.logical_not(tf.cast(padding_mask, tf.bool))
            self.padding_mask = padding_mask

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features, training = training)
        unmasked_features = self.dropout_features(
            unmasked_features, training = training
        )

        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None

        if self.input_quantizer:
            q = self.input_quantizer(features, produce_targets = False)
            features = q['x']
            num_vars = q['num_vars']
            code_ppl = q['code_perplexity']
            prob_ppl = q['prob_perplexity']
            curr_temp = q['temp']
            features = self.project_inp(features)

        if mask:
            x, mask_indices = self.apply_mask(
                features,
                padding_mask,
                mask_indices = mask_indices,
                mask_channel_indices = mask_channel_indices,
            )
            self.x = x
            y = unmasked_features
        else:
            x = features
            y = unmasked_features
            mask_indices = None

        x = self.encoder(x, padding_mask, training = training)

        if features_only:
            return {'x': x, 'padding_mask': padding_mask}

        if self.quantizer:
            q = self.quantizer(y, produce_targets = False)
            y = q['x']
            num_vars = q['num_vars']
            code_ppl = q['code_perplexity']
            prob_ppl = q['prob_perplexity']
            curr_temp = q['temp']

            y = self.project_q(y)

            negs, _ = self.sample_negatives(
                y, tf.shape(y)[1], padding_count = padding_count
            )

        else:
            y = self.project_q(y)
            negs, _ = self.sample_negatives(
                y, tf.shape(y)[1], padding_count = padding_count
            )

        if self.target_glu:
            y = self.target_glu(y)
            negs = self.target_glu(negs)

        x = self.final_proj(x)
        x = self.compute_preds(x, y, negs)
        result = {
            'x': x,
            'padding_mask': padding_mask,
            'features_pen': features_pen,
        }

        if prob_ppl is not None:
            result['prob_perplexity'] = prob_ppl
            result['code_perplexity'] = code_ppl

        return result, float(num_vars), curr_temp
