# -*- coding: utf-8 -*-
# Copyright 2020 The FastSpeech2 Authors and Minh Nguyen (@dathudeptrai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tensorflow Model modules for FastPitch."""

import tensorflow as tf
from ..fastspeech.layer import get_initializer
from ..fastspeech.model import Model as FastSpeech
from ..fastspeech2.model import FastSpeechVariantPredictor


class Model(FastSpeech):
    """TF Fastpitch module."""

    def __init__(self, config, **kwargs):
        """Init layers for fastspeech."""
        super().__init__(config, **kwargs)

        self.pitch_predictor = FastSpeechVariantPredictor(
            config, dtype=tf.float32, name='pitch_predictor'
        )
        self.duration_predictor = FastSpeechVariantPredictor(
            config, dtype=tf.float32, name='duration_predictor'
        )

        self.pitch_embeddings = tf.keras.layers.Conv1D(
            filters=config.encoder_self_attention_params.hidden_size,
            kernel_size=3,
            padding='same',
            name='pitch_embeddings',
        )

    def call(
        self,
        input_ids,
        duration_gts,
        pitch_gts,
        training=True,
        **kwargs,
    ):
        speaker_ids = tf.convert_to_tensor([0], tf.int32)
        attention_mask = tf.math.not_equal(input_ids, 0)
        embedding_output = self.embeddings(
            [input_ids, speaker_ids], training=training
        )
        encoder_output = self.encoder(
            [embedding_output, attention_mask], training=training
        )
        last_encoder_hidden_states = encoder_output[0]

        # energy predictor, here use last_encoder_hidden_states, u can use more hidden_states layers
        # rather than just use last_hidden_states of encoder for energy_predictor.
        duration_outputs = self.duration_predictor(
            [last_encoder_hidden_states, speaker_ids, attention_mask]
        )  # [batch_size, length]

        pitch_outputs = self.pitch_predictor(
            [last_encoder_hidden_states, speaker_ids, attention_mask],
            training=training,
        )

        pitch_embedding = self.pitch_embeddings(
            tf.expand_dims(pitch_gts, 2)
        )

        # sum features
        last_encoder_hidden_states += pitch_embedding

        length_regulator_outputs, encoder_masks = self.length_regulator(
            [last_encoder_hidden_states, duration_gts], training=training
        )

        # create decoder positional embedding
        decoder_pos = tf.range(
            1, tf.shape(length_regulator_outputs)[1] + 1, dtype=tf.int32
        )
        masked_decoder_pos = tf.expand_dims(decoder_pos, 0) * encoder_masks

        decoder_output = self.decoder(
            [
                length_regulator_outputs,
                speaker_ids,
                encoder_masks,
                masked_decoder_pos,
            ],
            training=training,
        )
        last_decoder_hidden_states = decoder_output[0]

        # here u can use sum or concat more than 1 hidden states layers from decoder.
        mels_before = self.mel_dense(last_decoder_hidden_states)
        mels_after = (
            self.postnet([mels_before, encoder_masks], training=training)
            + mels_before
        )

        outputs = (
            mels_before,
            mels_after,
            duration_outputs,
            pitch_outputs,
        )
        return outputs

    def inference(
        self, input_ids, speed_ratios, pitch_ratios, pitch_addition, **kwargs
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
        pitch_ratios = tf.expand_dims(pitch_ratios, 1)
        pitch_addition = tf.expand_dims(pitch_addition, 1)

        # energy predictor, here use last_encoder_hidden_states, u can use more hidden_states layers
        # rather than just use last_hidden_states of encoder for energy_predictor.
        duration_outputs = self.duration_predictor(
            [last_encoder_hidden_states, speaker_ids, attention_mask]
        )  # [batch_size, length]
        duration_outputs = tf.nn.relu(tf.math.exp(duration_outputs) - 1.0)
        duration_outputs = tf.cast(
            tf.math.round(duration_outputs * speed_ratios), tf.int32
        )

        pitch_outputs = self.pitch_predictor(
            [last_encoder_hidden_states, speaker_ids, attention_mask],
            training=False,
        )
        pitch_outputs = pitch_outputs + pitch_addition
        pitch_outputs = pitch_outputs * pitch_ratios

        pitch_embedding = self.pitch_embeddings(tf.expand_dims(pitch_outputs, 2))

        # sum features
        last_encoder_hidden_states += pitch_embedding

        length_regulator_outputs, encoder_masks = self.length_regulator(
            [last_encoder_hidden_states, duration_outputs], training=False
        )

        # create decoder positional embedding
        decoder_pos = tf.range(
            1, tf.shape(length_regulator_outputs)[1] + 1, dtype=tf.int32
        )
        masked_decoder_pos = tf.expand_dims(decoder_pos, 0) * encoder_masks

        decoder_output = self.decoder(
            [
                length_regulator_outputs,
                speaker_ids,
                encoder_masks,
                masked_decoder_pos,
            ],
            training=False,
        )
        last_decoder_hidden_states = decoder_output[0]

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
            pitch_outputs,
        )
        return outputs
