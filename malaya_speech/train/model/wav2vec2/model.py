import tensorflow as tf
from dataclasses import dataclass, field
from .layer import gelu, ConvFeatureExtractionModel


class Model(tf.keras.Model):
    def __init__(self, cfg, **kwargs):
        super(Model, self).__init__(name = 'wav2vec2', **kwargs)
        self.cfg = cfg

        feature_enc_layers = cfg.conv_feature_layers
        self.embed = feature_enc_layers[-1][0]
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers = feature_enc_layers,
            dropout = 0.0,
            mode = cfg.extractor_mode,
            conv_bias = cfg.conv_bias,
        )
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def _get_feat_extract_output_lengths(self, input_lengths):
        def _conv_out_length(input_length, kernel_size, stride):
            return tf.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = self.cfg.conv_feature_layers

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(
                input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2]
            )

        return tf.cast(input_lengths, tf.int32)

    def call(
        self,
        source,
        padding_mask = None,
        mask = True,
        features_only = False,
        training = True,
    ):
        features = self.feature_extractor(source, training = training)
        if self.feature_grad_mult < 0:
            features = tf.stop_gradient(features)

        features_pen = tf.reduce_mean(tf.math.pow(features, 2))
        features = self.layer_norm(features, training = training)
        if padding_mask is not None:
            input_lengths = padding_mask
            max_length = tf.cast(tf.reduce_max(padding_mask), tf.int32)
            padding_mask = tf.sequence_mask(
                lengths = padding_mask, maxlen = max_length, dtype = tf.float32
            )
            padding_mask.set_shape((None, None))
            padding_mask = tf.math.logical_not(padding_mask)

            output_lengths = self._get_feat_extract_output_lengths(
                input_lengths
            )
