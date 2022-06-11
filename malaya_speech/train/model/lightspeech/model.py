import tensorflow as tf
import math
import numpy as np
from ..melgan.layer import GroupConv1D
from ..fastspeech.model import (
    TFFastSpeechEmbeddings,
    TFFastSpeechAttention,
    TFFastSpeechOutput,
    TFTacotronPostnet,
    TFFastSpeechLengthRegulator,
    TFEmbedding
)
from ..fastspeech.layer import ACT2FN
from ..fastspeech2.model import FastSpeechVariantPredictor

# https://github.com/microsoft/NeuralSpeech/blob/master/LightSpeech/modules/operations.py#L559
# OPERATIONS_ENCODER = {
#     1: lambda : EncSepConvLayer(hparams['hidden_size'], 1, hparams['dropout'], hparams['activation']),  # h, num_heads, dropout
#     2: lambda : EncSepConvLayer(hparams['hidden_size'], 5, hparams['dropout'], hparams['activation']),
#     3: lambda : EncSepConvLayer(hparams['hidden_size'], 9, hparams['dropout'], hparams['activation']),
#     4: lambda : EncSepConvLayer(hparams['hidden_size'], 13, hparams['dropout'], hparams['activation']),
#     5: lambda : EncSepConvLayer(hparams['hidden_size'], 17, hparams['dropout'], hparams['activation']),
#     6: lambda : EncSepConvLayer(hparams['hidden_size'], 21, hparams['dropout'], hparams['activation']),
#     7: lambda : EncSepConvLayer(hparams['hidden_size'], 25, hparams['dropout'], hparams['activation']),
#     8: lambda : EncTransformerAttnLayer(hparams['hidden_size'], 2, hparams['dropout']),
#     9: lambda : EncTransformerAttnLayer(hparams['hidden_size'], 4, hparams['dropout']),
#     10: lambda : EncTransformerAttnLayer(hparams['hidden_size'], 8, hparams['dropout']),
#     11: lambda : EncTransformerFFNLayer(hparams['hidden_size'], hparams['filter_size'], hparams['ffn_kernel_size'], hparams['dropout'], hparams['activation']),
#     12: lambda : IdentityLayer(),
# }
# arch: '2 7 4 3 5 6 3 4'


class IdentityLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, training=True):
        return inputs


class ConvSeparable(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * out_channels))

        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D((self.kernel_size, 1),
                                                              padding=padding, use_bias=False,
                                                              depthwise_initializer=tf.keras.initializers.RandomNormal(
                                                                  mean=0.0, stddev=std, seed=None
        ),
        )
        self.pointwise_conv = tf.keras.layers.Conv1D(out_channels, 1, padding=padding,
                                                     kernel_initializer=tf.keras.initializers.RandomNormal(
                                                         mean=0.0, stddev=std, seed=None
                                                     ),
                                                     bias_initializer='zeros')

    def call(self, x, training=True):
        x = self.depthwise_conv(tf.expand_dims(x, 2))[:, :, 0]
        x = self.pointwise_conv(x)
        return x


class ConvSeparable_v2(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * out_channels))

        self.depthwise_conv = tf.keras.layers.Conv1D(in_channels, self.kernel_size,
                                                     padding=padding, use_bias=False,
                                                     kernel_initializer=tf.keras.initializers.RandomNormal(
                                                         mean=0.0, stddev=std, seed=None
                                                     ),
                                                     )
        self.pointwise_conv = tf.keras.layers.Conv1D(out_channels, 1, padding=padding,
                                                     kernel_initializer=tf.keras.initializers.RandomNormal(
                                                         mean=0.0, stddev=std, seed=None
                                                     ),
                                                     bias_initializer='zeros')

    def call(self, x, training=True):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class EncSepConvLayer(tf.keras.layers.Layer):
    def __init__(self, c, kernel_size, dropout, activation, **kwargs):
        super().__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.activation_fn = ACT2FN[activation]
        self.conv1 = ConvSeparable(c, c, kernel_size, dropout=dropout)
        self.conv2 = ConvSeparable(c, c, kernel_size, dropout=dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, inputs, training=False):
        hidden_states, attention_mask = inputs
        residual = hidden_states
        x = self.layer_norm(hidden_states)
        x = x * tf.cast(
            tf.expand_dims(attention_mask, 2), dtype=x.dtype
        )
        x = self.activation_fn(self.conv1(x))
        x = self.dropout(x, training=training)

        x = self.activation_fn(self.conv2(x))
        x = self.dropout(x, training=training)
        x = residual + x
        return x


class LightSpeechLayer(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.attention = TFFastSpeechAttention(config, name='attention')
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, inputs, training=False):
        """Call logic."""
        hidden_states, attention_mask = inputs

        attention_outputs = self.attention(
            [hidden_states, attention_mask], training=training
        )
        layer_output = self.layer_norm(attention_outputs[0])
        masked_layer_output = layer_output * tf.cast(
            tf.expand_dims(attention_mask, 2), dtype=layer_output.dtype
        )
        outputs = (masked_layer_output,) + attention_outputs[1:]
        return outputs


class Encoder(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        # 2 7 4 3
        self.layer1 = EncSepConvLayer(config.hidden_size, 5, config.hidden_dropout_prob, config.hidden_act)
        self.layer2 = EncSepConvLayer(config.hidden_size, 25, config.hidden_dropout_prob, config.hidden_act)
        self.layer3 = EncSepConvLayer(config.hidden_size, 13, config.hidden_dropout_prob, config.hidden_act)
        self.layer4 = EncSepConvLayer(config.hidden_size, 9, config.hidden_dropout_prob, config.hidden_act)

    def call(self, inputs, training=False):
        hidden_states, attention_mask = inputs
        o = self.layer1((hidden_states, attention_mask), training=training)
        o = self.layer2((o, attention_mask), training=training)
        o = self.layer3((o, attention_mask), training=training)
        o = self.layer4((o, attention_mask), training=training)
        return o


class Decoder(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 5 6 3 4
        self.config = config

        # create decoder positional embedding
        self.decoder_positional_embeddings = TFEmbedding(
            config.max_position_embeddings + 1,
            config.hidden_size,
            weights=[self._sincos_embedding()],
            name='position_embeddings',
            trainable=False,
        )
        self.layer1 = EncSepConvLayer(config.hidden_size, 17, config.hidden_dropout_prob, config.hidden_act)
        self.layer2 = EncSepConvLayer(config.hidden_size, 21, config.hidden_dropout_prob, config.hidden_act)
        self.layer3 = EncSepConvLayer(config.hidden_size, 9, config.hidden_dropout_prob, config.hidden_act)
        self.layer4 = EncSepConvLayer(config.hidden_size, 3, config.hidden_dropout_prob, config.hidden_act)

    def call(self, inputs, training=False):
        hidden_states, attention_mask, decoder_pos = inputs

        # calculate new hidden states.
        hidden_states += tf.cast(
            self.decoder_positional_embeddings(decoder_pos), hidden_states.dtype
        )
        o = self.layer1((hidden_states, attention_mask), training=training)
        o = self.layer2((o, attention_mask), training=training)
        o = self.layer3((o, attention_mask), training=training)
        o = self.layer4((o, attention_mask), training=training)
        return o

    def _sincos_embedding(self):
        position_enc = np.array(
            [
                [
                    pos
                    / np.power(10000, 2.0 * (i // 2) / self.config.hidden_size)
                    for i in range(self.config.hidden_size)
                ]
                for pos in range(self.config.max_position_embeddings + 1)
            ]
        )

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])

        # pad embedding.
        position_enc[0] = 0.0

        return position_enc


class Model(tf.keras.Model):

    def __init__(self, config, **kwargs):
        """Init layers for fastspeech."""
        super().__init__(**kwargs)
        self.embeddings = TFFastSpeechEmbeddings(config, name='embeddings')
        self.encoder = Encoder(config.encoder_self_attention_params, name='encoder')
        self.decoder = Decoder(config.decoder_self_attention_params, name='decoder')
        self.f0_predictor = FastSpeechVariantPredictor(
            config, dtype=tf.float32, name='f0_predictor'
        )
        self.energy_predictor = FastSpeechVariantPredictor(
            config, dtype=tf.float32, name='energy_predictor'
        )
        self.duration_predictor = FastSpeechVariantPredictor(
            config, dtype=tf.float32, name='duration_predictor'
        )

        # define f0_embeddings and energy_embeddings
        self.f0_embeddings = tf.keras.layers.Conv1D(
            filters=config.encoder_self_attention_params.hidden_size,
            kernel_size=9,
            padding='same',
            name='f0_embeddings',
        )
        self.f0_dropout = tf.keras.layers.Dropout(0.5)
        self.energy_embeddings = tf.keras.layers.Conv1D(
            filters=config.encoder_self_attention_params.hidden_size,
            kernel_size=9,
            padding='same',
            name='energy_embeddings',
        )
        self.energy_dropout = tf.keras.layers.Dropout(0.5)
        self.mel_dense = tf.keras.layers.Dense(
            units=config.num_mels, dtype=tf.float32, name='mel_before'
        )
        self.postnet = TFTacotronPostnet(
            config=config, dtype=tf.float32, name='postnet'
        )

        self.length_regulator = TFFastSpeechLengthRegulator(
            config, name='length_regulator'
        )

    def call(
        self,
        input_ids,
        duration_gts,
        f0_gts,
        energy_gts,
        training=True,
        **kwargs,
    ):
        speaker_ids = tf.convert_to_tensor([0], tf.int32)
        attention_mask = tf.math.not_equal(input_ids, 0)
        embedding_output = self.embeddings(
            [input_ids, speaker_ids], training=training
        )
        last_encoder_hidden_states = self.encoder((embedding_output, attention_mask),
                                                  training=training)

        duration_outputs = self.duration_predictor(
            [last_encoder_hidden_states, speaker_ids, attention_mask]
        )  # [batch_size, length]

        f0_outputs = self.f0_predictor(
            [last_encoder_hidden_states, speaker_ids, attention_mask],
            training=training,
        )
        energy_outputs = self.energy_predictor(
            [last_encoder_hidden_states, speaker_ids, attention_mask],
            training=training,
        )

        f0_embedding = self.f0_embeddings(
            tf.expand_dims(f0_gts, 2)
        )
        energy_embedding = self.energy_embeddings(
            tf.expand_dims(energy_gts, 2)
        )

        # apply dropout both training/inference
        f0_embedding = self.f0_dropout(f0_embedding, training=True)
        energy_embedding = self.energy_dropout(
            energy_embedding, training=True
        )

        last_encoder_hidden_states += f0_embedding + energy_embedding

        length_regulator_outputs, encoder_masks = self.length_regulator(
            [last_encoder_hidden_states, duration_gts], training=training
        )

        # create decoder positional embedding
        decoder_pos = tf.range(
            1, tf.shape(length_regulator_outputs)[1] + 1, dtype=tf.int32
        )
        masked_decoder_pos = tf.expand_dims(decoder_pos, 0) * encoder_masks

        last_decoder_hidden_states = self.decoder(
            [
                length_regulator_outputs,
                encoder_masks,
                masked_decoder_pos,
            ],
            training=training,
        )

        mels_before = self.mel_dense(last_decoder_hidden_states)
        mels_after = (
            self.postnet([mels_before, encoder_masks], training=training)
            + mels_before
        )

        outputs = (
            mels_before,
            mels_after,
            duration_outputs,
            f0_outputs,
            energy_outputs,
        )
        return outputs

    def inference(
        self, input_ids, speed_ratios, f0_ratios, energy_ratios, **kwargs
    ):
        speaker_ids = tf.convert_to_tensor([0], tf.int32)
        attention_mask = tf.math.not_equal(input_ids, 0)
        embedding_output = self.embeddings(
            [input_ids, speaker_ids], training=False
        )
        encoder_output = self.encoder(
            [embedding_output, attention_mask], training=False
        )
        last_encoder_hidden_states = encoder_output[0]

        # expand ratios
        speed_ratios = tf.expand_dims(speed_ratios, 1)  # [B, 1]
        f0_ratios = tf.expand_dims(f0_ratios, 1)  # [B, 1]
        energy_ratios = tf.expand_dims(energy_ratios, 1)  # [B, 1]

        # energy predictor, here use last_encoder_hidden_states, u can use more hidden_states layers
        # rather than just use last_hidden_states of encoder for energy_predictor.
        duration_outputs = self.duration_predictor(
            [last_encoder_hidden_states, speaker_ids, attention_mask]
        )  # [batch_size, length]
        duration_outputs = tf.nn.relu(tf.math.exp(duration_outputs) - 1.0)
        duration_outputs = tf.cast(
            tf.math.round(duration_outputs * speed_ratios), tf.int32
        )

        f0_outputs = self.f0_predictor(
            [last_encoder_hidden_states, speaker_ids, attention_mask],
            training=False,
        )
        f0_outputs *= f0_ratios

        energy_outputs = self.energy_predictor(
            [last_encoder_hidden_states, speaker_ids, attention_mask],
            training=False,
        )
        energy_outputs *= energy_ratios

        f0_embedding = self.f0_dropout(
            self.f0_embeddings(tf.expand_dims(f0_outputs, 2)), training=True
        )
        energy_embedding = self.energy_dropout(
            self.energy_embeddings(tf.expand_dims(energy_outputs, 2)),
            training=True,
        )

        # sum features
        last_encoder_hidden_states += f0_embedding + energy_embedding

        length_regulator_outputs, encoder_masks = self.length_regulator(
            [last_encoder_hidden_states, duration_outputs], training=False
        )

        # create decoder positional embedding
        decoder_pos = tf.range(
            1, tf.shape(length_regulator_outputs)[1] + 1, dtype=tf.int32
        )
        masked_decoder_pos = tf.expand_dims(decoder_pos, 0) * encoder_masks

        last_decoder_hidden_states = self.decoder(
            [
                length_regulator_outputs,
                encoder_masks,
                masked_decoder_pos,
            ],
            training=False,
        )

        # here u can use sum or concat more than 1 hidden states layers from decoder.
        mel_before = self.mel_dense(last_decoder_hidden_states)
        mel_after = (
            self.postnet([mel_before, encoder_masks], training=True)
            + mel_before
        )

        outputs = (
            mel_before,
            mel_after,
            duration_outputs,
            f0_outputs,
            energy_outputs,
        )
        return outputs
