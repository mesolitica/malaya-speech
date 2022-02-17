import tensorflow as tf
from ..wav2vec2.layer import ConvFeatureExtractionModel
from ..utils import shape_list
from ..masking import compute_mask_indices


class Model(tf.keras.Model):
    def __init__(self, cfg, encoder, **kwargs):
        super(Model, self).__init__(name='best_rq', **kwargs)
        self.cfg = cfg

        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )
        self.layer_norm = tf.keras.layers.LayerNormalization()

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

        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(cfg.embedding_dim, cfg.num_embeddings), dtype='float32'
            ),
            trainable=False,
            name='embeddings_vqvae',
        )
        self.projection = tf.Variable(
            initial_value=w_init(
                shape=(self.embed, cfg.embedding_dim), dtype='float32'
            ),
            trainable=False,
            name='projection_vqvae',
        )

        self.project_inp = tf.keras.layers.Dense(cfg.encoder_embed_dim)
        self.mask_emb = tf.get_variable(
            name='mask_emb',
            shape=[cfg.encoder_embed_dim],
            initializer=tf.truncated_normal_initializer(),
        )
        self.logits = tf.keras.layers.Dense(cfg.num_embeddings)
        self.encoder = encoder

    def get_code_indices(self, x):
        B, T, C = shape_list(x)
        flattened = tf.reshape(x, [-1, C])
        flattened_inputs = tf.matmul(flattened, self.projection)
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return tf.reshape(encoding_indices, [B, T])

    def _get_feat_extract_output_lengths(self, input_lengths):
        def _conv_out_length(input_length, kernel_size, stride):
            return tf.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = eval(self.cfg.conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(
                input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2]
            )

        return tf.cast(input_lengths, tf.int32)

    def apply_mask(self, x):
        B, T, C = shape_list(x)

        if self.mask_prob > 0:
            mask_time_indices = compute_mask_indices(
                (B, T),
                mask_prob=self.mask_prob,
                mask_length=self.mask_length,
                min_masks=2,
            )
            mask_time_indices = tf.cast(mask_time_indices[:, :, tf.newaxis], tf.bool)
            mask_time_indices = tf.tile(mask_time_indices, [1, 1, C])
            mask_emb_ = self.mask_emb[tf.newaxis, tf.newaxis, :]
            mask_emb_ = tf.tile(mask_emb_, [B, T, 1])
            x = tf.where(
                mask_time_indices,
                mask_emb_,
                x,
            )

        if self.mask_channel_prob > 0:
            mask_feature_indices = compute_mask_indices(
                (B, C),
                mask_prob=self.mask_channel_prob,
                mask_length=self.mask_channel_length,
            )
            mask_feature_indices = tf.cast(mask_feature_indices[:, tf.newaxis, :], tf.bool)
            mask_feature_indices = tf.tile(mask_feature_indices, [1, T, 1])
            zeros = tf.zeros(shape=(B, T, C))
            x = tf.where(mask_feature_indices, zeros, x)
        return x

    def call(
        self,
        source,
        padding_mask=None,
        mask=True,
        features_only=False,
        mask_channel_indices=None,
        padding_count=None,
        training=True,
    ):
        features = self.feature_extractor(source, training=training)
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

        features = self.dropout_input(features, training=training)
        features = self.project_inp(features)

        y = unmasked_features

        if mask:
            x = self.apply_mask(features)
        else:
            x = features

        self.x = x
        self.y = y

        x = self.encoder(x, padding_mask, training=training)
        if features_only:
            return {'x': x, 'padding_mask': padding_mask, 'y': y}
        else:
            onehot = self.get_code_indices(y)
            return {'x': self.logits(x), 'padding_mask': padding_mask, 'y': y, 'onehot': onehot}
