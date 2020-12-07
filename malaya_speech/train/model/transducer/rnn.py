# Copyright 2020 Huy Le Nguyen (@usimarit)
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
""" https://arxiv.org/pdf/1811.06621.pdf """

import tensorflow as tf
from .layer import Embedding, get_rnn, get_shape_invariants
import collections

Hypothesis = collections.namedtuple(
    'Hypothesis', ('index', 'prediction', 'states')
)
BLANK = 0


class TransducerPrediction(tf.keras.Model):
    def __init__(
        self,
        vocabulary_size: int,
        embed_dim: int,
        embed_dropout: float = 0,
        num_rnns: int = 1,
        rnn_units: int = 512,
        rnn_type: str = 'lstm',
        layer_norm: bool = True,
        projection_units: int = 0,
        kernel_regularizer = None,
        bias_regularizer = None,
        name = 'transducer_prediction',
        **kwargs,
    ):
        super(TransducerPrediction, self).__init__(name = name, **kwargs)
        self.embed = Embedding(
            vocabulary_size,
            embed_dim,
            regularizer = kernel_regularizer,
            name = f'{name}_embedding',
        )
        self.do = tf.keras.layers.Dropout(
            embed_dropout, name = f'{name}_dropout'
        )
        # Initialize rnn layers
        RNN = get_rnn(rnn_type)
        self.rnns = []
        for i in range(num_rnns):
            rnn = RNN(
                units = rnn_units,
                return_sequences = True,
                name = f'{name}_lstm_{i}',
                return_state = True,
                kernel_regularizer = kernel_regularizer,
                bias_regularizer = bias_regularizer,
            )
            if layer_norm:
                ln = tf.keras.layers.LayerNormalization(name = f'{name}_ln_{i}')
            else:
                ln = None
            if projection_units > 0:
                projection = tf.keras.layers.Dense(
                    projection_units,
                    name = f'{name}_projection_{i}',
                    kernel_regularizer = kernel_regularizer,
                    bias_regularizer = bias_regularizer,
                )
            else:
                projection = None
            self.rnns.append({'rnn': rnn, 'ln': ln, 'projection': projection})

    def get_initial_state(self):
        """Get zeros states
        Returns:
            tf.Tensor: states having shape [num_rnns, 1 or 2, B, P]
        """
        states = []
        for rnn in self.rnns:
            states.append(
                tf.stack(
                    rnn['rnn'].get_initial_state(
                        tf.zeros([1, 1, 1], dtype = tf.float32)
                    ),
                    axis = 0,
                )
            )
        return tf.stack(states, axis = 0)

    def call(self, inputs, training = False):
        outputs = self.embed(inputs, training = training)
        outputs = self.do(outputs, training = training)
        for rnn in self.rnns:
            outputs = rnn['rnn'](outputs, training = training)
            outputs = outputs[0]
            if rnn['ln'] is not None:
                outputs = rnn['ln'](outputs, training = training)
            if rnn['projection'] is not None:
                outputs = rnn['projection'](outputs, training = training)
        return outputs

    def recognize(self, inputs, states, training = False):
        """Recognize function for prediction network
        Args:
            inputs (tf.Tensor): shape [1, 1]
            states (tf.Tensor): shape [num_lstms, 2, B, P]
        Returns:
            tf.Tensor: outputs with shape [1, 1, P]
            tf.Tensor: new states with shape [num_lstms, 2, 1, P]
        """
        outputs = self.embed(inputs, training = training)
        outputs = self.do(outputs, training = training)
        new_states = []
        for i, rnn in enumerate(self.rnns):
            outputs = rnn['rnn'](
                outputs,
                training = training,
                initial_state = tf.unstack(states[i], axis = 0),
            )
            new_states.append(tf.stack(outputs[1:]))
            outputs = outputs[0]
            if rnn['ln'] is not None:
                outputs = rnn['ln'](outputs, training = False)
            if rnn['projection'] is not None:
                outputs = rnn['projection'](outputs, training = False)
        return outputs, tf.stack(new_states, axis = 0)

    def get_config(self):
        conf = self.embed.get_config()
        conf.update(self.do.get_config())
        for rnn in self.rnns:
            conf.update(rnn['rnn'].get_config())
            if rnn['ln'] is not None:
                conf.update(rnn['ln'].get_config())
            if rnn['projection'] is not None:
                conf.update(rnn['projection'].get_config())
        return conf


class TransducerJoint(tf.keras.Model):
    def __init__(
        self,
        vocabulary_size: int,
        joint_dim: int = 1024,
        kernel_regularizer = None,
        bias_regularizer = None,
        name = 'tranducer_joint',
        **kwargs,
    ):
        super(TransducerJoint, self).__init__(name = name, **kwargs)
        self.ffn_enc = tf.keras.layers.Dense(
            joint_dim,
            name = f'{name}_enc',
            kernel_regularizer = kernel_regularizer,
            bias_regularizer = bias_regularizer,
        )
        self.ffn_pred = tf.keras.layers.Dense(
            joint_dim,
            use_bias = False,
            name = f'{name}_pred',
            kernel_regularizer = kernel_regularizer,
        )
        self.ffn_out = tf.keras.layers.Dense(
            vocabulary_size,
            name = f'{name}_vocab',
            kernel_regularizer = kernel_regularizer,
            bias_regularizer = bias_regularizer,
        )

    def call(self, inputs, training = False):
        # enc has shape [B, T, E]
        # pred has shape [B, U, P]
        enc_out, pred_out = inputs
        enc_out = self.ffn_enc(
            enc_out, training = training
        )  # [B, T, E] => [B, T, V]
        pred_out = self.ffn_pred(
            pred_out, training = training
        )  # [B, U, P] => [B, U, V]
        enc_out = tf.expand_dims(enc_out, axis = 2)
        pred_out = tf.expand_dims(pred_out, axis = 1)
        outputs = tf.nn.tanh(enc_out + pred_out)  # => [B, T, U, V]
        outputs = self.ffn_out(outputs, training = training)
        return outputs

    def get_config(self):
        conf = self.ffn_enc.get_config()
        conf.update(self.ffn_pred.get_config())
        conf.update(self.ffn_out.get_config())
        return conf


class Model(tf.keras.Model):
    def __init__(
        self,
        encoder,
        vocabulary_size: int,
        embed_dim: int = 512,
        embed_dropout: float = 0,
        num_rnns: int = 1,
        rnn_units: int = 320,
        rnn_type: str = 'lstm',
        layer_norm: bool = False,
        projection_units: int = 0,
        joint_dim: int = 1024,
        kernel_regularizer = None,
        bias_regularizer = None,
        name = 'transducer',
        **kwargs,
    ):
        super(Model, self).__init__(name = name, **kwargs)
        self.encoder = encoder
        self.vocabulary_size = vocabulary_size
        self.predict_net = TransducerPrediction(
            vocabulary_size = vocabulary_size,
            embed_dim = embed_dim,
            embed_dropout = embed_dropout,
            num_rnns = num_rnns,
            rnn_units = rnn_units,
            rnn_type = rnn_type,
            layer_norm = layer_norm,
            projection_units = projection_units,
            kernel_regularizer = kernel_regularizer,
            bias_regularizer = bias_regularizer,
            name = f'{name}_prediction',
        )
        self.joint_net = TransducerJoint(
            vocabulary_size = vocabulary_size,
            joint_dim = joint_dim,
            kernel_regularizer = kernel_regularizer,
            bias_regularizer = bias_regularizer,
            name = f'{name}_joint',
        )

    def call(self, inputs, training = False):
        """
        Transducer Model call function
        Args:
            features: audio features in shape [B, T, F, C]
            predicted: predicted sequence of character ids, in shape [B, U]
            training: python boolean
        Returns:
            `logits` with shape [B, T, U, vocab]
        """
        features, predicted = inputs
        enc = self.encoder(features, training = training)
        pred = self.predict_net(predicted, training = training)
        outputs = self.joint_net([enc, pred], training = training)
        return outputs

    def encoder_inference(self, features):
        """
        Infer function for encoder (or encoders)
        Args:
            features (tf.Tensor): features with shape [B, T, F, C]
        Returns:
            tf.Tensor: output of encoders with shape [B, T, E]
        """
        outputs = tf.expand_dims(features, axis = 0)
        outputs = self.encoder(outputs, training = False)
        return tf.squeeze(outputs, axis = 0)

    def decoder_inference(self, encoded, predicted, states, training = False):
        """Infer function for decoder
        Args:
            encoded (tf.Tensor): output of encoder at each time step => shape [E]
            predicted (tf.Tensor): last character index of predicted sequence => shape []
            states (nested lists of tf.Tensor): states returned by rnn layers
        Returns:
            (ytu, new_states)
        """
        encoded = tf.reshape(encoded, [1, 1, -1])
        predicted = tf.reshape(predicted, [1, 1])
        y, new_states = self.predict_net.recognize(
            predicted, states, training = training
        )
        ytu = tf.nn.log_softmax(
            self.joint_net([encoded, y], training = training)
        )
        ytu = tf.squeeze(ytu, axis = None)
        return ytu, new_states

    def get_config(self):
        conf = self.encoder.get_config()
        conf.update(self.predict_net.get_config())
        conf.update(self.joint_net.get_config())
        return conf

    def greedy_decoder(self, features):
        total = tf.shape(features)[0]
        batch = tf.constant(0, dtype = tf.int32)
        decoded = tf.zeros(shape = (0, tf.shape(features)[1]), dtype = tf.int32)

        def condition(batch, total, features, decoded):
            return tf.less(batch, total)

        def body(batch, total, features, decoded):
            yseq = self.perform_greedy(
                features[batch],
                predicted = tf.constant(BLANK, dtype = tf.int32),
                states = self.predict_net.get_initial_state(),
                swap_memory = True,
            )
            yseq = tf.expand_dims(yseq.prediction, axis = 0)
            padding = [[0, 0], [0, tf.shape(features)[1] - tf.shape(yseq)[1]]]
            yseq = tf.pad(yseq, padding, 'CONSTANT')
            decoded = tf.concat([decoded, yseq], axis = 0)
            return batch + 1, total, features, decoded

        batch, total, features, decoded = tf.while_loop(
            condition,
            body,
            loop_vars = (batch, total, features, decoded),
            swap_memory = True,
            shape_invariants = (
                batch.get_shape(),
                total.get_shape(),
                get_shape_invariants(features),
                tf.TensorShape([None, None]),
            ),
        )

        return decoded

    def perform_greedy(self, features, predicted, states, swap_memory = False):
        encoded = self.encoder_inference(features)
        prediction = tf.TensorArray(
            dtype = tf.int32,
            size = (tf.shape(encoded)[0] + 1),
            dynamic_size = False,
            element_shape = tf.TensorShape([]),
            clear_after_read = False,
        )
        time = tf.constant(0, dtype = tf.int32)
        total = tf.shape(encoded)[0]

        hypothesis = Hypothesis(
            index = tf.constant(0, dtype = tf.int32),
            prediction = prediction.write(0, predicted),
            states = states,
        )

        def condition(time, total, encoded, hypothesis):
            return tf.less(time, total)

        def body(time, total, encoded, hypothesis):
            ytu, new_states = self.decoder_inference(
                encoded = tf.gather_nd(
                    encoded, tf.expand_dims(time, axis = -1)
                ),
                predicted = hypothesis.prediction.read(hypothesis.index),
                states = hypothesis.states,
            )
            char = tf.argmax(ytu, axis = -1, output_type = tf.int32)
            index, char, new_states = tf.cond(
                tf.equal(char, BLANK),
                true_fn = lambda: (
                    hypothesis.index + 1,
                    BLANK,
                    hypothesis.states,
                ),
                false_fn = lambda: (hypothesis.index + 1, char, new_states),
            )
            hypothesis = Hypothesis(
                index = index,
                prediction = hypothesis.prediction.write(index, char),
                states = new_states,
            )
            return time + 1, total, encoded, hypothesis

        time, total, encoded, hypothesis = tf.while_loop(
            condition,
            body,
            loop_vars = (time, total, encoded, hypothesis),
            swap_memory = swap_memory,
        )
        hypothesis = Hypothesis(
            index = hypothesis.index,
            prediction = tf.gather_nd(
                params = hypothesis.prediction.stack(),
                indices = tf.expand_dims(
                    tf.range(hypothesis.index + 1), axis = -1
                ),
            ),
            states = hypothesis.states,
        )

        return hypothesis
