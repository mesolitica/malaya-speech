import tensorflow as tf
from dataclasses import dataclass, field
from .layer import gelu, glu, ConvFeatureExtractionModel
from .masking import compute_mask_indices, index_put, index_put_constant
from ..utils import shape_list


class Model(tf.keras.Model):
    def __init__(self, cfg, encoder, dictionaries, **kwargs):
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

        if any([d is None for d in dictionaries]):
            print('cannot find dictionary. assume will be used for fine-tuning')
        else:
            self.num_classes = [len(d) for d in dictionaries]
            self.label_embs_concat = tf.get_variable(
                name='label_embs_concat',
                shape=[sum(self.num_classes), final_dim],
                initializer=tf.truncated_normal_initializer(),
            )

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
            y = torch.index_select(label_embs, 0, target.long())

        if self.quantizer:
            q = self.quantizer(y, produce_targets=False, training=training)
            y = q['x']
            num_vars = q['num_vars']
            code_ppl = q['code_perplexity']
            prob_ppl = q['prob_perplexity']
            curr_temp = q['temp']

            y = self.project_q(y)

            negs, _ = self.sample_negatives(
                y, tf.shape(y)[1], padding_count=padding_count
            )

        else:
            y = self.project_q(y)
            negs, _ = self.sample_negatives(
                y, tf.shape(y)[1], padding_count=padding_count
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
