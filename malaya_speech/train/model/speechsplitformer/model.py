import tensorflow as tf
from ..transformer import attention_layer
from ..transformer import ffn_layer
from ..transformer import model_utils
from ..transformer.transformer import (
    LayerNormalization,
    PrePostProcessingWrapper,
)
from ..speechsplit.model import InterpLnr
from ..transformer.transformer import DecoderStack
from ..transformer.model_utils import (
    get_position_encoding,
    get_padding_bias,
    get_decoder_self_attention_bias,
)
import copy


class Encoder_6(tf.layers.Layer):
    def __init__(self, params, hparams, train):
        super(Encoder_6, self).__init__()
        self.dim_neck_3 = hparams.dim_neck_3
        self.freq_3 = hparams.freq_3
        self.dim_f0 = hparams.dim_f0
        self.dim_enc_3 = hparams.dim_enc_3
        self.dim_emb = hparams.dim_spk_emb
        self.chs_grp = hparams.chs_grp
        self.layers = []

        self.before_dense_1 = tf.keras.layers.Dense(
            units=self.dim_enc_3, dtype=tf.float32, name='before_dense_1'
        )

        params = copy.deepcopy(params)
        self.params = params
        params['hidden_size'] = self.dim_enc_3
        for _ in range(params['num_hidden_layers']):
            self_attention_layer = attention_layer.SelfAttention(
                params['hidden_size'],
                params['num_heads'],
                params['attention_dropout'],
                train,
            )
            feed_forward_network = ffn_layer.FeedFowardNetwork(
                params['hidden_size'],
                params['filter_size'],
                params['relu_dropout'],
                train,
                params['allow_ffn_pad'],
                activation=params.get('activation', 'relu'),
            )

            self.layers.append(
                [
                    PrePostProcessingWrapper(
                        self_attention_layer, params, train
                    ),
                    PrePostProcessingWrapper(
                        feed_forward_network, params, train
                    ),
                ]
            )

        self.output_normalization = LayerNormalization(params['hidden_size'])
        self.encoder_dense_1 = tf.keras.layers.Dense(
            units=self.dim_neck_3,
            dtype=tf.float32,
            name='encoder_dense_1',
        )

        self.interp = InterpLnr(hparams)

    def call(self, encoder_inputs, attention_mask, training=True):
        encoder_inputs = self.before_dense_1(encoder_inputs)
        initializer = tf.variance_scaling_initializer(
            self.params['initializer_gain'],
            mode='fan_avg',
            distribution='uniform',
        )
        inputs_padding = tf.cast(
            tf.logical_not(tf.cast(attention_mask, tf.bool)), tf.float32
        )
        with tf.variable_scope('Transformer', initializer=initializer):
            attention_bias = get_padding_bias(inputs_padding)
            with tf.name_scope('encode'):

                if training:
                    encoder_inputs = tf.nn.dropout(
                        encoder_inputs,
                        1 - self.params['layer_postprocess_dropout'],
                    )

                for n, layer in enumerate(self.layers):
                    self_attention_layer = layer[0]
                    feed_forward_network = layer[1]

                    with tf.variable_scope('layer_%d' % n):
                        with tf.variable_scope('self_attention'):
                            encoder_inputs = self_attention_layer(
                                encoder_inputs, attention_bias
                            )
                        with tf.variable_scope('ffn'):
                            encoder_inputs = feed_forward_network(
                                encoder_inputs, inputs_padding
                            )

                    encoder_inputs = self.interp(
                        encoder_inputs,
                        tf.tile(
                            [tf.shape(encoder_inputs)[1]],
                            [tf.shape(encoder_inputs)[0]],
                        ),
                        training=True,
                    )

                return self.encoder_dense_1(
                    self.output_normalization(encoder_inputs)
                )


class Encoder_7(tf.layers.Layer):
    def __init__(self, params, hparams, train):
        super(Encoder_7, self).__init__()
        self.dim_neck = hparams.dim_neck
        self.dim_neck_3 = hparams.dim_neck_3
        self.dim_freq = hparams.dim_freq
        self.dim_enc = hparams.dim_enc
        self.dim_enc_3 = hparams.dim_enc_3
        self.layers_1, self.layers_2 = [], []
        self.params = params

        self.before_dense_1 = tf.keras.layers.Dense(
            units=self.dim_enc, dtype=tf.float32, name='before_dense_1'
        )
        self.before_dense_2 = tf.keras.layers.Dense(
            units=self.dim_enc_3, dtype=tf.float32, name='before_dense_2'
        )

        params_1 = copy.deepcopy(params)
        params_1['hidden_size'] = self.dim_enc
        params_2 = copy.deepcopy(params)
        params_2['hidden_size'] = self.dim_enc_3

        for _ in range(params_1['num_hidden_layers']):
            self_attention_layer = attention_layer.SelfAttention(
                params_1['hidden_size'],
                params_1['num_heads'],
                params_1['attention_dropout'],
                train,
            )
            feed_forward_network = ffn_layer.FeedFowardNetwork(
                params_1['hidden_size'],
                params_1['filter_size'],
                params_1['relu_dropout'],
                train,
                params_1['allow_ffn_pad'],
                activation=params_1.get('activation', 'relu'),
            )

            self.layers_1.append(
                [
                    PrePostProcessingWrapper(
                        self_attention_layer, params_1, train
                    ),
                    PrePostProcessingWrapper(
                        feed_forward_network, params_1, train
                    ),
                ]
            )

            self_attention_layer = attention_layer.SelfAttention(
                params_2['hidden_size'],
                params_2['num_heads'],
                params_2['attention_dropout'],
                train,
            )
            feed_forward_network = ffn_layer.FeedFowardNetwork(
                params_2['hidden_size'],
                params_2['filter_size'],
                params_2['relu_dropout'],
                train,
                params_2['allow_ffn_pad'],
                activation=params_2.get('activation', 'relu'),
            )

            self.layers_2.append(
                [
                    PrePostProcessingWrapper(
                        self_attention_layer, params_2, train
                    ),
                    PrePostProcessingWrapper(
                        feed_forward_network, params_2, train
                    ),
                ]
            )

        self.output_normalization_1 = LayerNormalization(
            params_1['hidden_size']
        )
        self.output_normalization_2 = LayerNormalization(
            params_2['hidden_size']
        )

        self.encoder_dense_1 = tf.keras.layers.Dense(
            units=self.dim_neck, dtype=tf.float32, name='encoder_dense_1'
        )
        self.encoder_dense_2 = tf.keras.layers.Dense(
            units=self.dim_neck_3,
            dtype=tf.float32,
            name='encoder_dense_2',
        )

        self.interp = InterpLnr(hparams)

    def call(self, encoder_inputs, attention_mask, training=True):
        initializer = tf.variance_scaling_initializer(
            self.params['initializer_gain'],
            mode='fan_avg',
            distribution='uniform',
        )
        inputs_padding = tf.cast(
            tf.logical_not(tf.cast(attention_mask, tf.bool)), tf.float32
        )
        with tf.variable_scope('Transformer', initializer=initializer):
            attention_bias = get_padding_bias(inputs_padding)
            with tf.name_scope('encode'):

                if training:
                    encoder_inputs = tf.nn.dropout(
                        encoder_inputs,
                        1 - self.params['layer_postprocess_dropout'],
                    )

                x = encoder_inputs[:, :, : self.dim_freq]
                f0 = encoder_inputs[:, :, self.dim_freq:]

                x = self.before_dense_1(x)
                f0 = self.before_dense_2(f0)

                for n, layer in enumerate(self.layers_1):
                    self_attention_layer = layer[0]
                    feed_forward_network = layer[1]

                    with tf.variable_scope('layer_1_%d' % n):
                        with tf.variable_scope('self_attention'):
                            x = self_attention_layer(x, attention_bias)
                        with tf.variable_scope('ffn'):
                            x = feed_forward_network(x, inputs_padding)

                    self_attention_layer = self.layers_2[n][0]
                    feed_forward_network = self.layers_2[n][1]

                    with tf.variable_scope('layer_2_%d' % n):
                        with tf.variable_scope('self_attention'):
                            f0 = self_attention_layer(f0, attention_bias)
                        with tf.variable_scope('ffn'):
                            f0 = feed_forward_network(f0, inputs_padding)

                    x_f0 = tf.concat((x, f0), axis=2)
                    x_f0 = self.interp(
                        x_f0,
                        tf.tile([tf.shape(x_f0)[1]], [tf.shape(x)[0]]),
                        training=True,
                    )
                    x = x_f0[:, :, : self.dim_enc]
                    f0 = x_f0[:, :, self.dim_enc:]

                x = x_f0[:, :, : self.dim_enc]
                f0 = x_f0[:, :, self.dim_enc:]
                x = self.encoder_dense_1(self.output_normalization_1(x))
                f0 = self.encoder_dense_2(self.output_normalization_2(f0))

                return x, f0


class Encoder_t(tf.layers.Layer):
    def __init__(self, params, hparams, train):
        super(Encoder_t, self).__init__()
        self.dim_neck_2 = hparams.dim_neck_2
        self.freq_2 = hparams.freq_2
        self.dim_freq = hparams.dim_freq
        self.dim_enc_2 = hparams.dim_enc_2
        self.dim_emb = hparams.dim_spk_emb
        self.chs_grp = hparams.chs_grp
        self.params = params

        self.before_dense = tf.keras.layers.Dense(
            units=self.dim_enc_2, dtype=tf.float32, name='before_dense_1'
        )

        params = copy.deepcopy(params)
        params['num_hidden_layers'] = 1
        params['hidden_size'] = self.dim_enc_2

        self.layers = []
        for _ in range(params['num_hidden_layers']):
            self_attention_layer = attention_layer.SelfAttention(
                params['hidden_size'],
                params['num_heads'],
                params['attention_dropout'],
                train,
            )
            feed_forward_network = ffn_layer.FeedFowardNetwork(
                params['hidden_size'],
                params['filter_size'],
                params['relu_dropout'],
                train,
                params['allow_ffn_pad'],
                activation=params.get('activation', 'relu'),
            )

            self.layers.append(
                [
                    PrePostProcessingWrapper(
                        self_attention_layer, params, train
                    ),
                    PrePostProcessingWrapper(
                        feed_forward_network, params, train
                    ),
                ]
            )

        self.output_normalization = LayerNormalization(params['hidden_size'])
        self.encoder_dense = tf.keras.layers.Dense(
            units=self.dim_neck_2, dtype=tf.float32, name='encoder_dense'
        )

    def call(self, encoder_inputs, attention_mask, training=True):
        encoder_inputs = self.before_dense(encoder_inputs)
        initializer = tf.variance_scaling_initializer(
            self.params['initializer_gain'],
            mode='fan_avg',
            distribution='uniform',
        )
        inputs_padding = tf.cast(
            tf.logical_not(tf.cast(attention_mask, tf.bool)), tf.float32
        )
        with tf.variable_scope('Transformer', initializer=initializer):
            attention_bias = get_padding_bias(inputs_padding)
            with tf.name_scope('encode'):

                if training:
                    encoder_inputs = tf.nn.dropout(
                        encoder_inputs,
                        1 - self.params['layer_postprocess_dropout'],
                    )
                for n, layer in enumerate(self.layers):
                    self_attention_layer = layer[0]
                    feed_forward_network = layer[1]

                    with tf.variable_scope('layer_%d' % n):
                        with tf.variable_scope('self_attention'):
                            encoder_inputs = self_attention_layer(
                                encoder_inputs, attention_bias
                            )
                        with tf.variable_scope('ffn'):
                            encoder_inputs = feed_forward_network(
                                encoder_inputs, inputs_padding
                            )
                return self.encoder_dense(
                    self.output_normalization(encoder_inputs)
                )


class Decoder_3(tf.keras.layers.Layer):
    def __init__(self, params, hparams, train, **kwargs):
        super(Decoder_3, self).__init__(name='Decoder_3', **kwargs)
        self.decoder_stack = DecoderStack(params, train)
        params = copy.deepcopy(params)
        self.params = params
        self.before_dense = tf.keras.layers.Dense(
            units=params['hidden_size'],
            dtype=tf.float32,
            name='before_dense_1',
        )
        self.linear_projection = tf.keras.layers.Dense(
            units=hparams.dim_freq,
            dtype=tf.float32,
            name='self.linear_projection',
        )

    def call(self, x, attention_mask, encoder_outputs, training=True):
        x = self.before_dense(x)
        input_shape = tf.shape(x)

        initializer = tf.variance_scaling_initializer(
            self.params['initializer_gain'],
            mode='fan_avg',
            distribution='uniform',
        )

        inputs_padding = tf.cast(
            tf.logical_not(tf.cast(attention_mask, tf.bool)), tf.float32
        )

        with tf.variable_scope('Decoder_3', initializer=initializer):
            attention_bias = get_padding_bias(inputs_padding)

            with tf.name_scope('decode'):
                decoder_inputs = x
                with tf.name_scope('add_pos_encoding'):
                    length = tf.shape(decoder_inputs)[1]
                if training:
                    decoder_inputs = tf.nn.dropout(
                        decoder_inputs,
                        1 - self.params['layer_postprocess_dropout'],
                    )
                decoder_self_attention_bias = get_decoder_self_attention_bias(
                    length
                )
                return self.linear_projection(
                    self.decoder_stack(
                        decoder_inputs,
                        encoder_outputs,
                        decoder_self_attention_bias,
                        attention_bias,
                    )
                )


class Decoder_4(tf.keras.layers.Layer):
    def __init__(self, params, hparams, train, **kwargs):
        super(Decoder_4, self).__init__(name='Decoder_4', **kwargs)
        self.decoder_stack = DecoderStack(params, train)
        params = copy.deepcopy(params)
        self.params = params
        self.before_dense = tf.keras.layers.Dense(
            units=params['hidden_size'],
            dtype=tf.float32,
            name='before_dense_1',
        )
        self.linear_projection = tf.keras.layers.Dense(
            units=hparams.dim_f0,
            dtype=tf.float32,
            name='self.linear_projection',
        )

    def call(self, x, attention_mask, encoder_outputs, training=True):
        x = self.before_dense(x)
        input_shape = tf.shape(x)

        initializer = tf.variance_scaling_initializer(
            self.params['initializer_gain'],
            mode='fan_avg',
            distribution='uniform',
        )

        inputs_padding = tf.cast(
            tf.logical_not(tf.cast(attention_mask, tf.bool)), tf.float32
        )

        with tf.variable_scope('Decoder_4', initializer=initializer):
            attention_bias = get_padding_bias(inputs_padding)

            with tf.name_scope('decode'):
                decoder_inputs = x
                with tf.name_scope('add_pos_encoding'):
                    length = tf.shape(decoder_inputs)[1]
                if training:
                    decoder_inputs = tf.nn.dropout(
                        decoder_inputs,
                        1 - self.params['layer_postprocess_dropout'],
                    )
                decoder_self_attention_bias = get_decoder_self_attention_bias(
                    length
                )
                return self.linear_projection(
                    self.decoder_stack(
                        decoder_inputs,
                        encoder_outputs,
                        decoder_self_attention_bias,
                        attention_bias,
                    )
                )


class Model(tf.keras.Model):
    def __init__(
        self, params_encoder, params_decoder, hparams, train=True, **kwargs
    ):
        super(Model, self).__init__(name='speechsplit', **kwargs)
        self.encoder_1 = Encoder_7(params_encoder, hparams, train=train)
        self.encoder_2 = Encoder_t(params_encoder, hparams, train=train)
        self.decoder = Decoder_3(params_decoder, hparams, train=train)
        self.freq = hparams.freq
        self.freq_2 = hparams.freq_2
        self.freq_3 = hparams.freq_3

    def call(self, x_f0, x_org, c_trg, mel_lengths, training=True):
        max_length = tf.cast(tf.reduce_max(mel_lengths), tf.int32)
        attention_mask = tf.sequence_mask(
            lengths=mel_lengths, maxlen=max_length, dtype=tf.float32
        )
        attention_mask.set_shape((None, None))

        codes_x, codes_f0 = self.encoder_1(
            x_f0, attention_mask, training=training
        )
        codes_2 = self.encoder_2(x_org, attention_mask, training=training)
        code_exp_1 = codes_x
        code_exp_3 = codes_f0
        code_exp_2 = codes_2
        c_trg = tf.tile(tf.expand_dims(c_trg, 1), (1, tf.shape(x_f0)[1], 1))

        encoder_outputs = tf.concat(
            (code_exp_1, code_exp_2, code_exp_3, c_trg), axis=-1
        )
        mel_outputs = self.decoder(
            encoder_outputs,
            attention_mask,
            encoder_outputs,
            training=training,
        )

        return codes_x, codes_f0, codes_2, encoder_outputs, mel_outputs


class Model_F0(tf.keras.Model):
    def __init__(
        self, params_encoder, params_decoder, hparams, train=True, **kwargs
    ):
        super(Model_F0, self).__init__(name='speechsplit_f0', **kwargs)
        self.encoder_2 = Encoder_t(params_encoder, hparams, train=train)
        self.encoder_3 = Encoder_6(params_encoder, hparams, train=train)
        self.decoder = Decoder_4(params_decoder, hparams, train=train)
        self.freq_2 = hparams.freq_2
        self.freq_3 = hparams.freq_3

    def call(self, x_org, f0_trg, mel_lengths, training=True):
        max_length = tf.cast(tf.reduce_max(mel_lengths), tf.int32)
        attention_mask = tf.sequence_mask(
            lengths=mel_lengths, maxlen=max_length, dtype=tf.float32
        )
        attention_mask.set_shape((None, None))

        codes_2 = self.encoder_2(x_org, attention_mask, training=training)
        code_exp_2 = codes_2
        codes_3 = self.encoder_3(f0_trg, attention_mask, training=training)
        code_exp_3 = codes_3
        self.o = [code_exp_2, code_exp_3]
        encoder_outputs = tf.concat((code_exp_2, code_exp_3), axis=-1)
        mel_outputs = self.decoder(
            encoder_outputs,
            attention_mask,
            encoder_outputs,
            training=training,
        )
        return codes_2, codes_3, encoder_outputs, mel_outputs
