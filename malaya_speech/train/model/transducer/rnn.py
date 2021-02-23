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
from .layer import (
    Embedding,
    get_rnn,
    get_shape_invariants,
    pad_prediction_tfarray,
    count_non_blank,
)
from ..utils import shape_list
import collections

Hypothesis = collections.namedtuple(
    'Hypothesis', ('index', 'prediction', 'states')
)
BeamHypothesis = collections.namedtuple(
    'BeamHypothesis', ('score', 'indices', 'prediction', 'states')
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

    def greedy_decoder(
        self,
        features,
        encoded_length,
        parallel_iterations = 10,
        swap_memory = False,
        training = False,
    ):
        encoded = self.encoder(features, training = training)
        encoded_length = (
            encoded_length
            // self.encoder.conv_subsampling.time_reduction_factor
        )
        total = tf.shape(features)[0]
        batch = tf.constant(0, dtype = tf.int32)
        decoded = tf.zeros(shape = (0, tf.shape(features)[1]), dtype = tf.int32)

        def condition(batch, decoded):
            return tf.less(batch, total)

        def body(batch, decoded):
            hypothesis = self._perform_greedy(
                encoded = encoded[batch],
                encoded_length = encoded_length[batch],
                predicted = tf.constant(BLANK, dtype = tf.int32),
                states = self.predict_net.get_initial_state(),
                parallel_iterations = parallel_iterations,
                swap_memory = swap_memory,
            )
            yseq = tf.expand_dims(hypothesis.prediction, axis = 0)
            padding = [[0, 0], [0, tf.shape(features)[1] - tf.shape(yseq)[1]]]
            yseq = tf.pad(yseq, padding, 'CONSTANT')
            decoded = tf.concat([decoded, yseq], axis = 0)
            return batch + 1, decoded

        batch, decoded = tf.while_loop(
            condition,
            body,
            loop_vars = (batch, decoded),
            parallel_iterations = parallel_iterations,
            swap_memory = True,
            shape_invariants = (
                batch.get_shape(),
                tf.TensorShape([None, None]),
            ),
            back_prop = False,
        )

        return decoded

    def _perform_greedy(
        self,
        encoded,
        encoded_length,
        predicted,
        states,
        parallel_iterations = 10,
        swap_memory = False,
    ):
        time = tf.constant(0, dtype = tf.int32)
        total = encoded_length

        hypothesis = Hypothesis(
            index = predicted,
            prediction = tf.TensorArray(
                dtype = tf.int32,
                size = total,
                dynamic_size = False,
                clear_after_read = False,
                element_shape = tf.TensorShape([]),
            ),
            states = states,
        )

        def condition(_time, _hypothesis):
            return tf.less(_time, total)

        def body(_time, _hypothesis):
            ytu, _states = self.decoder_inference(
                encoded = tf.gather_nd(
                    encoded, tf.expand_dims(_time, axis = -1)
                ),
                predicted = _hypothesis.index,
                states = _hypothesis.states,
            )
            _predict = tf.argmax(ytu, axis = -1, output_type = tf.int32)
            _equal = tf.equal(_predict, BLANK)
            _index = tf.where(_equal, _hypothesis.index, _predict)
            _states = tf.where(_equal, _hypothesis.states, _states)

            _prediction = _hypothesis.prediction.write(_time, _predict)
            _hypothesis = Hypothesis(
                index = _index, prediction = _prediction, states = _states
            )

            return _time + 1, _hypothesis

        time, hypothesis = tf.while_loop(
            condition,
            body,
            loop_vars = (time, hypothesis),
            parallel_iterations = parallel_iterations,
            swap_memory = swap_memory,
            back_prop = False,
        )
        return Hypothesis(
            index = hypothesis.index,
            prediction = hypothesis.prediction.stack(),
            states = hypothesis.states,
        )

    def beam_decoder(
        self,
        features,
        encoded_length,
        beam_width = 5,
        norm_score = True,
        parallel_iterations = 10,
        swap_memory = False,
        training = False,
    ):
        encoded = self.encoder(features, training = training)
        encoded_length = (
            encoded_length
            // self.encoder.conv_subsampling.time_reduction_factor
        )
        total = tf.shape(features)[0]
        batch = tf.constant(0, dtype = tf.int32)
        decoded = tf.zeros(shape = (0, tf.shape(features)[1]), dtype = tf.int32)

        def condition(batch, decoded):
            return tf.less(batch, total)

        def body(batch, decoded):
            hypothesis = self._perform_beam(
                encoded = encoded[batch],
                encoded_length = encoded_length[batch],
                beam_width = beam_width,
                norm_score = norm_score,
                parallel_iterations = parallel_iterations,
                swap_memory = swap_memory,
            )
            yseq = tf.expand_dims(hypothesis.prediction, axis = 0)
            padding = [[0, 0], [0, tf.shape(features)[1] - tf.shape(yseq)[1]]]
            yseq = tf.pad(yseq, padding, 'CONSTANT')
            decoded = tf.concat([decoded, yseq], axis = 0)
            return batch + 1, decoded

        batch, decoded = tf.while_loop(
            condition,
            body,
            loop_vars = (batch, decoded),
            parallel_iterations = parallel_iterations,
            swap_memory = True,
            shape_invariants = (
                batch.get_shape(),
                tf.TensorShape([None, None]),
            ),
            back_prop = False,
        )

        return decoded

    def _perform_beam(
        self,
        encoded,
        encoded_length,
        beam_width = 5,
        norm_score = True,
        parallel_iterations = 10,
        swap_memory = True,
    ):
        total = encoded_length

        def initialize_beam(dynamic = False):
            return BeamHypothesis(
                score = tf.TensorArray(
                    dtype = tf.float32,
                    size = beam_width if not dynamic else 0,
                    dynamic_size = dynamic,
                    element_shape = tf.TensorShape([]),
                    clear_after_read = False,
                ),
                indices = tf.TensorArray(
                    dtype = tf.int32,
                    size = beam_width if not dynamic else 0,
                    dynamic_size = dynamic,
                    element_shape = tf.TensorShape([]),
                    clear_after_read = False,
                ),
                prediction = tf.TensorArray(
                    dtype = tf.int32,
                    size = beam_width if not dynamic else 0,
                    dynamic_size = dynamic,
                    element_shape = None,
                    clear_after_read = False,
                ),
                states = tf.TensorArray(
                    dtype = tf.float32,
                    size = beam_width if not dynamic else 0,
                    dynamic_size = dynamic,
                    element_shape = tf.TensorShape(
                        shape_list(self.predict_net.get_initial_state())
                    ),
                    clear_after_read = False,
                ),
            )

        B = initialize_beam()
        B = BeamHypothesis(
            score = B.score.write(0, 0.0),
            indices = B.indices.write(0, BLANK),
            prediction = B.prediction.write(
                0, tf.ones([total], dtype = tf.int32) * BLANK
            ),
            states = B.states.write(0, self.predict_net.get_initial_state()),
        )

        def condition(time, total, B):
            return tf.less(time, total)

        def body(time, total, B):
            A = initialize_beam(dynamic = True)
            A = BeamHypothesis(
                score = A.score.unstack(B.score.stack()),
                indices = A.indices.unstack(B.indices.stack()),
                prediction = A.prediction.unstack(
                    pad_prediction_tfarray(B.prediction, blank = BLANK).stack()
                ),
                states = A.states.unstack(B.states.stack()),
            )
            A_i = tf.constant(0, tf.int32)
            B = initialize_beam()

            encoded_t = tf.gather_nd(encoded, tf.expand_dims(time, axis = -1))

            def beam_condition(beam, beam_width, A, A_i, B):
                return tf.less(beam, beam_width)

            def beam_body(beam, beam_width, A, A_i, B):
                # get y_hat
                y_hat_score, y_hat_score_index = tf.math.top_k(
                    A.score.stack(), k = 1, sorted = True
                )
                y_hat_score = y_hat_score[0]
                y_hat_index = tf.gather_nd(A.indices.stack(), y_hat_score_index)
                y_hat_prediction = tf.gather_nd(
                    pad_prediction_tfarray(A.prediction, blank = BLANK).stack(),
                    y_hat_score_index,
                )
                y_hat_states = tf.gather_nd(A.states.stack(), y_hat_score_index)

                # remove y_hat from A
                remain_indices = tf.range(
                    0, tf.shape(A.score.stack())[0], dtype = tf.int32
                )
                remain_indices = tf.gather_nd(
                    remain_indices,
                    tf.where(
                        tf.not_equal(remain_indices, y_hat_score_index[0])
                    ),
                )
                remain_indices = tf.expand_dims(remain_indices, axis = -1)
                A = BeamHypothesis(
                    score = A.score.unstack(
                        tf.gather_nd(A.score.stack(), remain_indices)
                    ),
                    indices = A.indices.unstack(
                        tf.gather_nd(A.indices.stack(), remain_indices)
                    ),
                    prediction = A.prediction.unstack(
                        tf.gather_nd(
                            pad_prediction_tfarray(
                                A.prediction, blank = BLANK
                            ).stack(),
                            remain_indices,
                        )
                    ),
                    states = A.states.unstack(
                        tf.gather_nd(A.states.stack(), remain_indices)
                    ),
                )
                A_i = tf.cond(
                    tf.equal(A_i, 0),
                    true_fn = lambda: A_i,
                    false_fn = lambda: A_i - 1,
                )

                ytu, new_states = self.decoder_inference(
                    encoded = encoded_t,
                    predicted = y_hat_index,
                    states = y_hat_states,
                )

                def predict_condition(pred, A, A_i, B):
                    return tf.less(pred, self.vocabulary_size)

                def predict_body(pred, A, A_i, B):
                    new_score = y_hat_score + tf.gather_nd(
                        ytu, tf.expand_dims(pred, axis = -1)
                    )

                    def true_fn():
                        return (
                            B.score.write(beam, new_score),
                            B.indices.write(beam, y_hat_index),
                            B.prediction.write(beam, y_hat_prediction),
                            B.states.write(beam, y_hat_states),
                            A.score,
                            A.indices,
                            A.prediction,
                            A.states,
                            A_i,
                        )

                    def false_fn():
                        scatter_index = count_non_blank(
                            y_hat_prediction, blank = BLANK
                        )
                        updated_prediction = tf.tensor_scatter_nd_update(
                            y_hat_prediction,
                            indices = tf.reshape(scatter_index, [1, 1]),
                            updates = tf.expand_dims(pred, axis = -1),
                        )
                        return (
                            B.score,
                            B.indices,
                            B.prediction,
                            B.states,
                            A.score.write(A_i, new_score),
                            A.indices.write(A_i, pred),
                            A.prediction.write(A_i, updated_prediction),
                            A.states.write(A_i, new_states),
                            A_i + 1,
                        )

                    b_score, b_indices, b_prediction, b_states, a_score, a_indices, a_prediction, a_states, A_i = tf.cond(
                        tf.equal(pred, BLANK),
                        true_fn = true_fn,
                        false_fn = false_fn,
                    )

                    B = BeamHypothesis(
                        score = b_score,
                        indices = b_indices,
                        prediction = b_prediction,
                        states = b_states,
                    )
                    A = BeamHypothesis(
                        score = a_score,
                        indices = a_indices,
                        prediction = a_prediction,
                        states = a_states,
                    )

                    return pred + 1, A, A_i, B

                _, A, A_i, B = tf.while_loop(
                    predict_condition,
                    predict_body,
                    loop_vars = [0, A, A_i, B],
                    parallel_iterations = parallel_iterations,
                    swap_memory = swap_memory,
                    back_prop = False,
                )

                return beam + 1, beam_width, A, A_i, B

            _, _, A, A_i, B = tf.while_loop(
                beam_condition,
                beam_body,
                loop_vars = [0, beam_width, A, A_i, B],
                parallel_iterations = parallel_iterations,
                swap_memory = swap_memory,
                back_prop = False,
            )

            return time + 1, total, B

        _, _, B = tf.while_loop(
            condition,
            body,
            loop_vars = [0, total, B],
            parallel_iterations = parallel_iterations,
            swap_memory = swap_memory,
            back_prop = False,
        )

        scores = B.score.stack()
        prediction = pad_prediction_tfarray(B.prediction, blank = BLANK).stack()
        if norm_score:
            prediction_lengths = count_non_blank(
                prediction, blank = BLANK, axis = 1
            )
            scores /= tf.cast(prediction_lengths, dtype = scores.dtype)

        y_hat_score, y_hat_score_index = tf.math.top_k(scores, k = 1)
        y_hat_score = y_hat_score[0]
        y_hat_index = tf.gather_nd(B.indices.stack(), y_hat_score_index)
        y_hat_prediction = tf.gather_nd(prediction, y_hat_score_index)
        y_hat_states = tf.gather_nd(B.states.stack(), y_hat_score_index)

        return Hypothesis(
            index = y_hat_index,
            prediction = y_hat_prediction,
            states = y_hat_states,
        )
