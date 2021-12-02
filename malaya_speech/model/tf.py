import tensorflow as tf
import numpy as np
import collections
from malaya_speech.utils import featurization
from malaya_speech.model.frame import Frame
from malaya_speech.utils.padding import (
    sequence_nd as padding_sequence_nd,
    sequence_1d,
)
from malaya_speech.utils.char import CTC_VOCAB
from malaya_speech.utils.char import decode as char_decode
from malaya_speech.utils.subword import (
    decode as subword_decode,
    encode as subword_encode,
    decode_multilanguage,
    get_index_multilanguage,
    align_multilanguage,
)
from malaya_speech.utils.execute import execute_graph
from malaya_speech.utils.activation import softmax, apply_temp
from malaya_speech.utils.featurization import universal_mel
from malaya_speech.utils.read import resample
from malaya_speech.utils.speechsplit import (
    quantize_f0_numpy,
    get_f0_sptk,
    get_fo_pyworld,
)
from malaya_speech.utils.lm import (
    BeamHypothesis_LM,
    BeamHypothesis,
    sort_and_trim_beams,
    prune_history,
    get_lm_beams,
)
from malaya_speech.utils.constant import MEL_MEAN, MEL_STD


class Abstract:
    def __str__(self):
        return f'<{self.__name__}: {self.__model__}>'

    def _execute(self, inputs, input_labels, output_labels):
        return execute_graph(
            inputs=inputs,
            input_labels=input_labels,
            output_labels=output_labels,
            sess=self._sess,
            input_nodes=self._input_nodes,
            output_nodes=self._output_nodes,
        )


class Speakernet(Abstract):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        vectorizer,
        sess,
        model,
        extra,
        label,
        name,
    ):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._vectorizer = vectorizer
        self._sess = sess
        self._extra = extra
        self.labels = label
        self.__model__ = model
        self.__name__ = name

    def vectorize(self, inputs):
        """
        Vectorize inputs.

        Parameters
        ----------
        inputs: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].

        Returns
        -------
        result: np.array
            returned [B, D].
        """
        inputs = [
            input.array if isinstance(input, Frame) else input
            for input in inputs
        ]

        inputs = [self._vectorizer(input) for input in inputs]
        inputs, lengths = padding_sequence_nd(
            inputs, dim=0, return_len=True
        )

        r = self._execute(
            inputs=[inputs, lengths],
            input_labels=['Placeholder', 'Placeholder_1'],
            output_labels=['logits'],
        )
        return r['logits']

    def __call__(self, inputs):
        return self.vectorize(inputs)


class Speaker2Vec(Abstract):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        vectorizer,
        sess,
        model,
        extra,
        label,
        name,
    ):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._vectorizer = vectorizer
        self._sess = sess
        self._extra = extra
        self.labels = label
        self.__model__ = model
        self.__name__ = name

    def vectorize(self, inputs):
        """
        Vectorize inputs.

        Parameters
        ----------
        inputs: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].

        Returns
        -------
        result: np.array
            returned [B, D].
        """
        inputs = [
            input.array if isinstance(input, Frame) else input
            for input in inputs
        ]

        inputs = [self._vectorizer(input, **self._extra) for input in inputs]

        if self.__model__ == 'deep-speaker':
            dim = 0
        else:
            dim = 1
        inputs = padding_sequence_nd(inputs, dim=dim)
        inputs = np.expand_dims(inputs, -1)

        r = self._execute(
            inputs=[inputs],
            input_labels=['Placeholder'],
            output_labels=['logits'],
        )
        return r['logits']

    def __call__(self, inputs):
        return self.vectorize(inputs)


class SpeakernetClassification(Abstract):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        vectorizer,
        sess,
        model,
        extra,
        label,
        name,
    ):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._vectorizer = vectorizer
        self._sess = sess
        self._extra = extra
        self.labels = label
        self.__model__ = model
        self.__name__ = name

    def predict_proba(self, inputs):
        """
        Predict inputs, will return probability.

        Parameters
        ----------
        inputs: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].

        Returns
        -------
        result: np.array
            returned [B, D].
        """
        inputs = [
            input.array if isinstance(input, Frame) else input
            for input in inputs
        ]

        inputs = [self._vectorizer(input) for input in inputs]
        inputs, lengths = padding_sequence_nd(
            inputs, dim=0, return_len=True
        )

        r = self._execute(
            inputs=[inputs, lengths],
            input_labels=['Placeholder', 'Placeholder_1'],
            output_labels=['logits'],
        )
        return softmax(r['logits'], axis=-1)

    def predict(self, inputs):
        """
        Predict inputs, will return labels.

        Parameters
        ----------
        inputs: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].

        Returns
        -------
        result: List[str]
            returned [B].
        """
        probs = np.argmax(self.predict_proba(inputs), axis=1)
        return [self.labels[p] for p in probs]

    def __call__(self, input):
        """
        Predict input, will return label.

        Parameters
        ----------
        inputs: np.array
            np.array or malaya_speech.model.frame.Frame.

        Returns
        -------
        result: str
        """

        return self.predict([input])[0]


class Classification(Abstract):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        vectorizer,
        sess,
        model,
        extra,
        label,
        name,
    ):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._vectorizer = vectorizer
        self._sess = sess
        self._extra = extra
        self.labels = label
        self.__model__ = model
        self.__name__ = name

    def predict_proba(self, inputs):
        """
        Predict inputs, will return probability.

        Parameters
        ----------
        inputs: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].

        Returns
        -------
        result: np.array
            returned [B, D].
        """
        inputs = [
            input.array if isinstance(input, Frame) else input
            for input in inputs
        ]

        inputs = [self._vectorizer(input, **self._extra) for input in inputs]
        if self.__model__ == 'deep-speaker':
            dim = 0
        else:
            dim = 1

        inputs = padding_sequence_nd(inputs, dim=dim)
        inputs = np.expand_dims(inputs, -1)

        r = self._execute(
            inputs=[inputs],
            input_labels=['Placeholder'],
            output_labels=['logits'],
        )
        return softmax(r['logits'], axis=-1)

    def predict(self, inputs):
        """
        Predict inputs, will return labels.

        Parameters
        ----------
        inputs: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].

        Returns
        -------
        result: List[str]
            returned [B].
        """
        probs = np.argmax(self.predict_proba(inputs), axis=1)
        return [self.labels[p] for p in probs]

    def __call__(self, input):
        """
        Predict input, will return label.

        Parameters
        ----------
        inputs: np.array
            np.array or malaya_speech.model.frame.Frame.

        Returns
        -------
        result: str
        """
        return self.predict([input])[0]


class UNET(Abstract):
    def __init__(self, input_nodes, output_nodes, sess, model, name):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._sess = sess
        self.__model__ = model
        self.__name__ = name

    def predict(self, inputs):
        """
        Enhance inputs, will return melspectrogram.

        Parameters
        ----------
        inputs: List[np.array]

        Returns
        -------
        result: List
        """
        inputs = [
            input.array if isinstance(input, Frame) else input
            for input in inputs
        ]
        mels = [featurization.scale_mel(s).T for s in inputs]
        x, lens = padding_sequence_nd(
            mels, maxlen=256, dim=0, return_len=True
        )

        r = self._execute(
            inputs=[x],
            input_labels=['Placeholder'],
            output_labels=['logits'],
        )
        l = r['logits']

        results = []
        for index in range(len(x)):
            results.append(
                featurization.unscale_mel(
                    x[index, : lens[index]].T + l[index, : lens[index], :, 0].T
                )
            )
        return results

    def __call__(self, inputs):
        return self.predict(inputs)


class UNETSTFT(Abstract):
    def __init__(
        self, input_nodes, output_nodes, instruments, sess, model, name
    ):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._instruments = instruments
        self._sess = sess
        self.__model__ = model
        self.__name__ = name

    def predict(self, input):
        """
        Enhance inputs, will return waveform.

        Parameters
        ----------
        input: np.array
            np.array or malaya_speech.model.frame.Frame.

        Returns
        -------
        result: Dict
        """
        if isinstance(input, Frame):
            input = input.array

        r = self._execute(
            inputs=[input],
            input_labels=['Placeholder'],
            output_labels=list(self._output_nodes.keys()),
        )
        results = {}
        for no, instrument in enumerate(self._instruments):
            results[instrument] = r[f'logits_{no}']
        return results

    def __call__(self, input):
        """
        Enhance inputs, will return waveform.

        Parameters
        ----------
        input: np.array
            np.array or malaya_speech.model.frame.Frame.

        Returns
        -------
        result: Dict
        """
        return self.predict(input)


class UNET1D(Abstract):
    def __init__(self, input_nodes, output_nodes, sess, model, name):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._sess = sess
        self.__model__ = model
        self.__name__ = name

    def predict(self, input):
        """
        Enhance inputs, will return waveform.

        Parameters
        ----------
        input: np.array
            np.array or malaya_speech.model.frame.Frame.

        Returns
        -------
        result: np.array
        """
        if isinstance(input, Frame):
            input = input.array

        r = self._execute(
            inputs=[input],
            input_labels=['Placeholder'],
            output_labels=['logits'],
        )
        return r['logits']

    def __call__(self, input):
        """
        Enhance inputs, will return waveform.

        Parameters
        ----------
        input: np.array
            np.array or malaya_speech.model.frame.Frame.

        Returns
        -------
        result: np.array
        """
        return self.predict(input)


class Transducer(Abstract):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        featurizer,
        vocab,
        time_reduction_factor,
        sess,
        model,
        name,
        wavs,
        stack=False,
        dummy_sentences=None,
    ):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._back_pad = np.zeros(shape=(2000,))
        self._front_pad = np.zeros(shape=(200,))

        self._featurizer = featurizer
        self._vocab = vocab
        self._time_reduction_factor = time_reduction_factor
        self._sess = sess
        self.__model__ = model
        self.__name__ = name
        self._wavs = wavs
        self._stack = stack
        if self._stack:
            self._len_vocab = [l.vocab_size for l in self._vocab]
            self._vocabs = {}
            k = 0
            for v in self._vocab:
                for i in range(v.vocab_size - 1):
                    self._vocabs[k] = v._id_to_subword(i)
                    k += 1
        else:
            self._vocabs = {i: self._vocab._id_to_subword(i) for i in range(self._vocab.vocab_size - 1)}
        self._vocabs[-1] = ''

    def _check_decoder(self, decoder, beam_width):
        decoder = decoder.lower()
        if decoder not in ['greedy', 'beam']:
            raise ValueError('mode only supports [`greedy`, `beam`]')
        if beam_width < 1:
            raise ValueError('beam_width must bigger than 0')
        return decoder

    def _get_inputs(self, inputs):
        inputs = [
            input.array if isinstance(input, Frame) else input
            for input in inputs
        ]

        index = len(inputs)

        # pretty hacky, result from single batch is not good caused by batchnorm.
        # have to append extra random wavs
        if len(inputs) < len(self._wavs) + 1:
            inputs = inputs + self._wavs[:(len(self._wavs) + 1) - len(inputs)]

        # padded, lens = sequence_1d(inputs, return_len=True)
        # padded = np.concatenate([self._front_pad, padded, self._back_pad], axis=-1)
        # lens = [l + len(self._back_pad) + len(self._front_pad) for l in lens]

        inputs = [np.concatenate([self._front_pad, wav, self._back_pad], axis=-1) for wav in inputs]
        padded, lens = sequence_1d(inputs, return_len=True)

        return padded, lens, index

    def _combined_indices(
        self, subwords, ids, l, reduction_factor=160, sample_rate=16000
    ):
        result, temp_l, temp_r = [], [], []
        for i in range(len(subwords)):
            if ids[i] is None and len(temp_r):
                data = {
                    'text': ''.join(temp_l),
                    'start': round(temp_r[0], 4),
                    'end': round(
                        temp_r[-1] + (reduction_factor / sample_rate), 4
                    ),
                }
                result.append(data)
                temp_l, temp_r = [], []
            else:
                temp_l.append(subwords[i])
                temp_r.append(l[ids[i]])

        if len(temp_l):
            data = {
                'text': ''.join(temp_l),
                'start': round(temp_r[0], 4),
                'end': round(temp_r[-1] + (reduction_factor / sample_rate), 4),
            }
            result.append(data)

        return result

    def _beam_decoder_lm(self, enc, total, initial_states, language_model,
                         beam_width=5, token_min_logp=-20.0, beam_prune_logp=-5.0,
                         temperature=0.0, score_norm=True, **kwargs):
        kept_hyps = [
            BeamHypothesis_LM(score=0.0, score_lm=0.0,
                              prediction=[0], states=initial_states,
                              text='', next_word='', word_part='')
        ]
        cached_lm_scores = {
            '': (0.0, 0.0, language_model.get_start_state())
        }
        cached_p_lm_scores: Dict[str, float] = {}
        B = kept_hyps
        for i in range(total):
            A = B
            B = []
            while True:
                y_hat = max(A, key=lambda x: x.score)
                A.remove(y_hat)
                r = self._execute(
                    inputs=[enc[i], y_hat.prediction[-1], y_hat.states],
                    input_labels=[
                        'encoded_placeholder',
                        'predicted_placeholder',
                        'states_placeholder',
                    ],
                    output_labels=['ytu', 'new_states'],
                )
                ytu_, new_states_ = r['ytu'], r['new_states']
                if temperature > 0:
                    ytu_ = apply_temp(ytu_, temperature=temperature)
                B.append(BeamHypothesis_LM(
                    score=y_hat.score + ytu_[0],
                    score_lm=0.0,
                    prediction=y_hat.prediction,
                    states=y_hat.states,
                    text=y_hat.text,
                    next_word=y_hat.next_word,
                    word_part=y_hat.word_part,
                ))
                ytu_ = ytu_[1:]
                max_idx = ytu_.argmax()
                idx_list = set(np.where(ytu_ >= token_min_logp)[0]) | {max_idx}
                for k in idx_list:
                    w = self._vocabs[k]
                    if isinstance(w, bytes):
                        w = w.decode(encoding='ISO-8859-1')
                    w = w.replace('_', ' ')
                    s = y_hat.score + ytu_[k]
                    p = y_hat.prediction + [k + 1]

                    if w[-1] == ' ':
                        beam_hyp = BeamHypothesis_LM(
                            score=s,
                            score_lm=0.0,
                            prediction=p,
                            states=new_states_,
                            text=y_hat.text,
                            next_word=y_hat.word_part + w,
                            word_part=''
                        )
                    else:
                        beam_hyp = BeamHypothesis_LM(
                            score=s,
                            score_lm=0.0,
                            prediction=p,
                            states=new_states_,
                            text=y_hat.text,
                            next_word=y_hat.next_word,
                            word_part=y_hat.word_part + w
                        )
                    A.append(beam_hyp)

                scored_beams = get_lm_beams(A, cached_lm_scores, cached_p_lm_scores, language_model)
                max_beam = max(scored_beams, key=lambda x: x.score_lm)
                max_score = max_beam.score_lm
                scored_beams = [b for b in scored_beams if b.score_lm >= max_score + beam_prune_logp]
                trimmed_beams = sort_and_trim_beams(scored_beams, beam_width=beam_width)
                A = prune_history(trimmed_beams, lm_order=language_model.order)

                max_A = max(A, key=lambda x: x.score)
                hyps_max = max_A.score
                kept_most_prob = [hyp for hyp in B if hyp.score > hyps_max]
                if len(kept_most_prob) >= beam_width:
                    B = kept_most_prob
                    break

        new_beams = []
        for beam in B:
            text = beam.text
            next_word = beam.next_word
            word_part = beam.word_part
            last_char = beam.prediction[-1]
            logit_score = beam.score
            beam_hyp = BeamHypothesis_LM(
                score=logit_score,
                score_lm=0.0,
                prediction=beam.prediction,
                states=beam.states,
                text=text,
                next_word=word_part,
                word_part=''
            )
            new_beams.append(beam_hyp)

        scored_beams = get_lm_beams(new_beams, cached_lm_scores, cached_p_lm_scores, language_model, is_eos=True)
        max_beam = max(scored_beams, key=lambda x: x.score_lm)
        max_score = max_beam.score_lm
        scored_beams = [b for b in scored_beams if b.score_lm >= max_score + beam_prune_logp]
        trimmed_beams = sort_and_trim_beams(scored_beams, beam_width)

        if score_norm:
            trimmed_beams.sort(key=lambda x: x.score_lm / len(x.prediction), reverse=True)
        else:
            trimmed_beams.sort(key=lambda x: x.score_lm, reverse=True)
        return trimmed_beams[0].prediction

    def _beam_decoder(
        self, enc, total, initial_states,
        beam_width=10, temperature=0.0, score_norm=True, **kwargs
    ):
        kept_hyps = [
            BeamHypothesis(
                score=0.0, prediction=[0], states=initial_states
            )
        ]
        B = kept_hyps
        for i in range(total):
            A = B
            B = []
            while True:
                y_hat = max(A, key=lambda x: x.score)
                A.remove(y_hat)
                r = self._execute(
                    inputs=[enc[i], y_hat.prediction[-1], y_hat.states],
                    input_labels=[
                        'encoded_placeholder',
                        'predicted_placeholder',
                        'states_placeholder',
                    ],
                    output_labels=['ytu', 'new_states'],
                )
                ytu_, new_states_ = r['ytu'], r['new_states']
                if temperature > 0:
                    ytu_ = apply_temp(ytu_, temperature=temperature)
                top_k = ytu_[1:].argsort()[-beam_width:][::-1]
                B.append(BeamHypothesis(
                    score=y_hat.score + ytu_[0],
                    prediction=y_hat.prediction,
                    states=y_hat.states,
                ))
                for k in top_k:
                    beam_hyp = BeamHypothesis(
                        score=y_hat.score + ytu_[k + 1],
                        prediction=y_hat.prediction + [k + 1],
                        states=new_states_,
                    )
                    A.append(beam_hyp)
                hyps_max = max(A, key=lambda x: x.score).score
                kept_most_prob = sorted(
                    [hyp for hyp in B if hyp.score > hyps_max],
                    key=lambda x: x.score,
                )
                if len(kept_most_prob) >= beam_width:
                    B = kept_most_prob
                    break

        if score_norm:
            B.sort(key=lambda x: x.score / len(x.prediction), reverse=True)
        else:
            B.sort(key=lambda x: x.score, reverse=True)
        return B[0].prediction

    def _beam(self, inputs, language_model=None,
              beam_width=5,
              token_min_logp=-20.0,
              beam_prune_logp=-5.0,
              temperature=0.0,
              score_norm=True,
              **kwargs):
        padded, lens, index = self._get_inputs(inputs)
        results = []

        if language_model:
            beam_function = self._beam_decoder_lm
        else:
            beam_function = self._beam_decoder

        r = self._execute(
            inputs=[padded, lens],
            input_labels=['X_placeholder', 'X_len_placeholder'],
            output_labels=['encoded', 'padded_lens', 'initial_states'],
        )
        encoded_, padded_lens_, s = (
            r['encoded'],
            r['padded_lens'],
            r['initial_states'],
        )
        padded_lens_ = padded_lens_ // self._time_reduction_factor
        for i in range(index):
            r = beam_function(
                enc=encoded_[i],
                total=padded_lens_[i],
                initial_states=s,
                beam_width=beam_width,
                temperature=temperature,
                language_model=language_model,
                token_min_logp=token_min_logp,
                beam_prune_logp=beam_prune_logp,
                score_norm=score_norm,
            )
            if self._stack:
                d = decode_multilanguage(self._vocab, r)
            else:
                d = subword_decode(self._vocab, r)
            results.append(d)
        return results

    def predict_alignment(self, input, combined=True):
        """
        Transcribe input and get timestamp, only support greedy decoder.

        Parameters
        ----------
        input: np.array
            np.array or malaya_speech.model.frame.Frame.
        combined: bool, optional (default=True)
            If True, will combined subwords to become a word.

        Returns
        -------
        result: List[Dict[text, start, end]]
        """

        padded, lens, index = self._get_inputs([input])
        r = self._execute(
            inputs=[padded, lens],
            input_labels=['X_placeholder', 'X_len_placeholder'],
            output_labels=['non_blank_transcript', 'non_blank_stime'],
        )
        non_blank_transcript = r['non_blank_transcript']
        non_blank_stime = r['non_blank_stime']
        if combined:
            if self._stack:
                words, indices = align_multilanguage(
                    self._vocab, non_blank_transcript, get_index=True
                )
            else:
                words, indices = self._vocab.decode(
                    non_blank_transcript, get_index=True
                )
        else:
            words, indices = [], []
            for no, ids in enumerate(non_blank_transcript):
                if self._stack:
                    last_index, v = get_index_multilanguage(ids, self._vocab, self._len_vocab)
                    w = self._vocab[last_index]._id_to_subword(v - 1)
                else:
                    w = self._vocab._id_to_subword(ids - 1)
                if isinstance(w, bytes):
                    w = w.decode()
                words.extend([w, None])
                indices.extend([no, None])

        return self._combined_indices(words, indices, non_blank_stime)

    def greedy_decoder(self, inputs):
        """
        Transcribe inputs using greedy decoder.

        Parameters
        ----------
        inputs: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].

        Returns
        -------
        result: List[str]
        """
        padded, lens, index = self._get_inputs(inputs)
        results = []
        r = self._execute(
            inputs=[padded, lens],
            input_labels=['X_placeholder', 'X_len_placeholder'],
            output_labels=['greedy_decoder'],
        )['greedy_decoder']

        for row in r[:index]:
            if self._stack:
                d = decode_multilanguage(self._vocab, row[row > 0])
            else:
                d = subword_decode(self._vocab, row[row > 0])
            results.append(d)

        return results

    def beam_decoder(self, inputs, beam_width: int = 5,
                     temperature: float = 0.0,
                     score_norm: bool = True):
        """
        Transcribe inputs using beam decoder.

        Parameters
        ----------
        inputs: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].
        beam_width: int, optional (default=5)
            beam size for beam decoder.
        temperature: float, optional (default=0.0)
            apply temperature function for logits, can help for certain case,
            logits += -np.log(-np.log(uniform_noise_shape_logits)) * temperature
        score_norm: bool, optional (default=True)
            descending sort beam based on score / length of decoded.

        Returns
        -------
        result: List[str]
        """
        return self._beam(inputs=inputs,
                          beam_width=beam_width, temperature=temperature,
                          score_norm=score_norm)

    def beam_decoder_lm(self, inputs, language_model,
                        beam_width: int = 5,
                        token_min_logp: float = -20.0,
                        beam_prune_logp: float = -5.0,
                        temperature: float = 0.0,
                        score_norm: bool = True):
        """
        Transcribe inputs using beam decoder + KenLM.

        Parameters
        ----------
        inputs: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].
        language_model: pyctcdecode.language_model.LanguageModel
            pyctcdecode language model, load from `LanguageModel(kenlm_model, alpha = alpha, beta = beta)`.
        beam_width: int, optional (default=5)
            beam size for beam decoder.
        token_min_logp: float, optional (default=-20.0)
            minimum log probability to select a token.
        beam_prune_logp: float, optional (default=-5.0)
            filter candidates >= max score lm + `beam_prune_logp`.
        temperature: float, optional (default=0.0)
            apply temperature function for logits, can help for certain case,
            logits += -np.log(-np.log(uniform_noise_shape_logits)) * temperature
        score_norm: bool, optional (default=True)
            descending sort beam based on score / length of decoded.

        Returns
        -------
        result: List[str]
        """
        return self._beam(inputs=inputs, language_model=language_model,
                          beam_width=beam_width, token_min_logp=token_min_logp,
                          beam_prune_logp=beam_prune_logp, temperature=temperature,
                          score_norm=score_norm)

    def predict(self, inputs):
        """
        Transcribe inputs using greedy decoder, will return list of strings.

        Parameters
        ----------
        inputs: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].

        Returns
        -------
        result: List[str]
        """
        return self.greedy_decoder(inputs)

    def __call__(self, input):
        """
        Transcribe input using greedy decoder, will return a string.

        Parameters
        ----------
        input: np.array
            np.array or malaya_speech.model.frame.Frame.

        Returns
        -------
        result: str
        """
        return self.predict([input], decoder=decoder, **kwargs)[0]


class Vocoder(Abstract):
    def __init__(self, input_nodes, output_nodes, sess, model, name):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._sess = sess
        self.__model__ = model
        self.__name__ = name

    def predict(self, inputs):
        """
        Change Mel to Waveform.

        Parameters
        ----------
        inputs: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].
        Returns
        -------
        result: List
        """
        inputs = [
            input.array if isinstance(input, Frame) else input
            for input in inputs
        ]
        padded, lens = sequence_1d(inputs, return_len=True)

        r = self._execute(
            inputs=[padded],
            input_labels=['Placeholder'],
            output_labels=['logits'],
        )
        return r['logits'][:, :, 0]

    def __call__(self, input):
        return self.predict([input])[0]


class Tacotron(Abstract):
    def __init__(
        self, input_nodes, output_nodes, normalizer, stats, sess, model, name
    ):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._normalizer = normalizer
        self._stats = stats
        self._sess = sess
        self.__model__ = model
        self.__name__ = name

    def predict(self, string, **kwargs):
        """
        Change string to Mel.

        Parameters
        ----------
        string: str

        Returns
        -------
        result: Dict[string, decoder-output, postnet-output, universal-output, alignment]
        """

        t, ids = self._normalizer.normalize(string, **kwargs)
        r = self._execute(
            inputs=[[ids], [len(ids)]],
            input_labels=['Placeholder', 'Placeholder_1'],
            output_labels=[
                'decoder_output',
                'post_mel_outputs',
                'alignment_histories',
            ],
        )
        v = r['post_mel_outputs'][0] * self._stats[1] + self._stats[0]
        v = (v - MEL_MEAN) / MEL_STD
        return {
            'string': t,
            'ids': ids,
            'decoder-output': r['decoder_output'][0],
            'postnet-output': r['post_mel_outputs'][0],
            'universal-output': v,
            'alignment': r['alignment_histories'][0],
        }

    def __call__(self, input):
        return self.predict(input)


class Fastspeech(Abstract):
    def __init__(
        self, input_nodes, output_nodes, normalizer, stats, sess, model, name
    ):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._normalizer = normalizer
        self._stats = stats
        self._sess = sess
        self.__model__ = model
        self.__name__ = name

    def predict(
        self,
        string,
        speed_ratio: float = 1.0,
        f0_ratio: float = 1.0,
        energy_ratio: float = 1.0,
        **kwargs,
    ):
        """
        Change string to Mel.

        Parameters
        ----------
        string: str
        speed_ratio: float, optional (default=1.0)
            Increase this variable will increase time voice generated.
        f0_ratio: float, optional (default=1.0)
            Increase this variable will increase frequency, low frequency will generate more deeper voice.
        energy_ratio: float, optional (default=1.0)
            Increase this variable will increase loudness.

        Returns
        -------
        result: Dict[string, decoder-output, postnet-output, universal-output]
        """
        t, ids = self._normalizer.normalize(string, **kwargs)
        r = self._execute(
            inputs=[[ids], [speed_ratio], [f0_ratio], [energy_ratio]],
            input_labels=[
                'Placeholder',
                'speed_ratios',
                'f0_ratios',
                'energy_ratios',
            ],
            output_labels=['decoder_output', 'post_mel_outputs'],
        )
        v = r['post_mel_outputs'][0] * self._stats[1] + self._stats[0]
        v = (v - MEL_MEAN) / MEL_STD
        return {
            'string': t,
            'ids': ids,
            'decoder-output': r['decoder_output'][0],
            'postnet-output': r['post_mel_outputs'][0],
            'universal-output': v,
        }

    def __call__(self, input, **kwargs):
        return self.predict(input, **kwargs)


class FastVC(Abstract):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        speaker_vector,
        magnitude,
        sess,
        model,
        name,
    ):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._speaker_vector = speaker_vector
        self._magnitude = magnitude
        self._sess = sess
        self.__model__ = model
        self.__name__ = name

    def predict(self, original_audio, target_audio):
        """
        Change original voice audio to follow targeted voice.

        Parameters
        ----------
        original_audio: np.array or malaya_speech.model.frame.Frame
        target_audio: np.array or malaya_speech.model.frame.Frame

        Returns
        -------
        result: Dict[decoder-output, postnet-output]
        """
        original_audio = (
            input.array if isinstance(original_audio, Frame) else original_audio
        )
        target_audio = (
            input.array if isinstance(target_audio, Frame) else target_audio
        )

        original_mel = universal_mel(original_audio)
        target_mel = universal_mel(target_audio)

        original_v = self._magnitude(self._speaker_vector([original_audio])[0])
        target_v = self._magnitude(self._speaker_vector([target_audio])[0])

        r = self._execute(
            inputs=[
                [original_mel],
                [original_v],
                [target_v],
                [len(original_mel)],
            ],
            input_labels=[
                'mel',
                'ori_vector',
                'target_vector',
                'mel_lengths',
            ],
            output_labels=['mel_before', 'mel_after'],
        )
        return {
            'decoder-output': r['mel_before'][0],
            'postnet-output': r['mel_after'][0],
        }

    def __call__(self, original_audio, target_audio):
        return self.predict(original_audio, target_audio)


class Split_Wav(Abstract):
    def __init__(self, input_nodes, output_nodes, sess, model, name):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._sess = sess
        self.__model__ = model
        self.__name__ = name

    def predict(self, input):
        """
        Split an audio into 4 different speakers.

        Parameters
        ----------
        input: np.array or malaya_speech.model.frame.Frame

        Returns
        -------
        result: np.array
        """
        if isinstance(input, Frame):
            input = input.array

        r = self._execute(
            inputs=[np.expand_dims([input], axis=-1)],
            input_labels=['Placeholder'],
            output_labels=['logits'],
        )
        r = r['logits']
        return r[:, 0, :, 0]

    def __call__(self, input):
        return self.predict(input)


class Split_Mel(Abstract):
    def __init__(self, input_nodes, output_nodes, sess, model, name):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._sess = sess
        self.__model__ = model
        self.__name__ = name

    def _to_mel(self, y):
        mel = universal_mel(y)
        mel[mel <= np.log(1e-2)] = np.log(1e-2)
        return mel

    def predict(self, input):
        """
        Split an audio into 4 different speakers.

        Parameters
        ----------
        input: np.array or malaya_speech.model.frame.Frame

        Returns
        -------
        result: np.array
        """
        if isinstance(input, Frame):
            input = input.array

        input = self._to_mel(input)

        r = self._execute(
            inputs=[input],
            input_labels=['Placeholder', 'Placeholder_1'],
            output_labels=['logits'],
        )
        r = r['logits']
        return r[:, 0]

    def __call__(self, input):
        return self.predict(input)


class Wav2Vec2_CTC(Abstract):
    def __init__(self, input_nodes, output_nodes, sess, model, name):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._sess = sess
        self.__model__ = model
        self.__name__ = name
        self._beam_width = 0

    def _check_decoder(self, decoder, beam_width):
        decoder = decoder.lower()
        if decoder not in ['greedy', 'beam']:
            raise ValueError('mode only supports [`greedy`, `beam`]')
        if beam_width < 1:
            raise ValueError('beam_width must bigger than 0')
        return decoder

    def _get_logits(self, padded, lens):
        r = self._execute(
            inputs=[padded, lens],
            input_labels=['X_placeholder', 'X_len_placeholder'],
            output_labels=['logits', 'seq_lens'],
        )
        return r['logits'], r['seq_lens']

    def _tf_ctc(self, padded, lens, beam_width, **kwargs):
        if tf.executing_eagerly():
            logits, seq_lens = self._get_logits(padded, lens)
            decoded = tf.compat.v1.nn.ctc_beam_search_decoder(
                logits,
                seq_lens,
                beam_width=beam_width,
                top_paths=1,
                merge_repeated=True,
                **kwargs,
            )
            preds = tf.sparse.to_dense(tf.compat.v1.to_int32(decoded[0][0]))
        else:
            if beam_width != self._beam_width:
                self._beam_width = beam_width
                self._decoded = tf.compat.v1.nn.ctc_beam_search_decoder(
                    self._output_nodes['logits'],
                    self._output_nodes['seq_lens'],
                    beam_width=self._beam_width,
                    top_paths=1,
                    merge_repeated=True,
                    **kwargs,
                )[0][0]

            r = self._sess.run(
                self._decoded,
                feed_dict={
                    self._input_nodes['X_placeholder']: padded,
                    self._input_nodes['X_len_placeholder']: lens,
                },
            )
            preds = np.zeros(r.dense_shape, dtype=np.int32)
            for i in range(r.values.shape[0]):
                preds[r.indices[i][0], r.indices[i][1]] = r.values[i]
        return preds

    def _predict(
        self, inputs, decoder: str = 'beam', beam_width: int = 100, **kwargs
    ):

        decoder = self._check_decoder(decoder, beam_width)

        inputs = [
            input.array if isinstance(input, Frame) else input
            for input in inputs
        ]

        padded, lens = sequence_1d(inputs, return_len=True)

        if decoder == 'greedy':
            beam_width = 1

        decoded = self._tf_ctc(padded, lens, beam_width, **kwargs)

        results = []
        for i in range(len(decoded)):
            r = char_decode(decoded[i], lookup=CTC_VOCAB).replace(
                '<PAD>', ''
            )
            results.append(r)
        return results

    def greedy_decoder(self, inputs):
        """
        Transcribe inputs using greedy decoder.

        Parameters
        ----------
        input: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].

        Returns
        -------
        result: List[str]
        """
        return self._predict(inputs=inputs, decoder='greedy')

    def beam_decoder(self, inputs, beam_width: int = 100):
        """
        Transcribe inputs using beam decoder.

        Parameters
        ----------
        input: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].
        beam_width: int, optional (default=100)
            beam size for beam decoder.

        Returns
        -------
        result: List[str]
        """
        return self._predict(inputs=inputs, decoder='beam', beam_width=beam_width)

    def predict(self, inputs):
        """
        Predict logits from inputs using greedy decoder.

        Parameters
        ----------
        input: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].


        Returns
        -------
        result: List[str]
        """
        return self.greedy_decoder(inputs=inputs)

    def predict_logits(self, inputs):
        """
        Predict logits from inputs.

        Parameters
        ----------
        input: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].


        Returns
        -------
        result: List[np.array]
        """

        inputs = [
            input.array if isinstance(input, Frame) else input
            for input in inputs
        ]

        padded, lens = sequence_1d(inputs, return_len=True)
        logits, seq_lens = self._get_logits(padded, lens)
        logits = np.transpose(logits, axes=(1, 0, 2))
        logits = softmax(logits, axis=-1)
        results = []
        for i in range(len(logits)):
            results.append(logits[i][: seq_lens[i]])
        return results

    def __call__(self, input):
        """
        Transcribe input using greedy decoder.

        Parameters
        ----------
        input: np.array
            np.array or malaya_speech.model.frame.Frame.

        Returns
        -------
        result: str
        """
        return self.predict([input])[0]


class CTC(Abstract):
    def __init__(self, input_nodes, output_nodes, sess, model, name):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._sess = sess
        self.__model__ = model
        self.__name__ = name


class FastSpeechSplit(Abstract):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        speaker_vector,
        gender_model,
        sess,
        model,
        name,
    ):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._speaker_vector = speaker_vector
        self._gender_model = gender_model
        self._sess = sess
        self.__model__ = model
        self.__name__ = name
        self._modes = {'R', 'F', 'U', 'RF', 'RU', 'FU', 'RFU'}
        self._freqs = {'female': [100, 600], 'male': [50, 250]}

    def _get_data(self, x, sr=22050, target_sr=16000):
        x_16k = resample(x, sr, target_sr)
        if self._gender_model is not None:
            gender = self._gender_model(x_16k)
            lo, hi = self._freqs.get(gender, [50, 250])
            f0 = get_f0_sptk(x, lo, hi)
        else:
            f0 = get_fo_pyworld(x)
        f0 = np.expand_dims(f0, -1)
        mel = universal_mel(x)
        v = self._speaker_vector([x_16k])[0]
        v = v / v.max()

        if len(mel) > len(f0):
            mel = mel[: len(f0)]
        return x, mel, f0, v

    def predict(
        self,
        original_audio,
        target_audio,
        modes=['R', 'F', 'U', 'RF', 'RU', 'FU', 'RFU'],
    ):
        """
        Change original voice audio to follow targeted voice.

        Parameters
        ----------
        original_audio: np.array or malaya_speech.model.frame.Frame
        target_audio: np.array or malaya_speech.model.frame.Frame
        modes: List[str], optional (default = ['R', 'F', 'U', 'RF', 'RU', 'FU', 'RFU'])
            R denotes rhythm, F denotes pitch target, U denotes speaker target (vector).

            * ``'R'`` - maintain `original_audio` F and U on `target_audio` R.
            * ``'F'`` - maintain `original_audio` R and U on `target_audio` F.
            * ``'U'`` - maintain `original_audio` R and F on `target_audio` U.
            * ``'RF'`` - maintain `original_audio` U on `target_audio` R and F.
            * ``'RU'`` - maintain `original_audio` F on `target_audio` R and U.
            * ``'FU'`` - maintain `original_audio` R on `target_audio` F and U.
            * ``'RFU'`` - no conversion happened, just do encoder-decoder on `target_audio`

        Returns
        -------
        result: Dict[modes]
        """
        s = set(modes) - self._modes
        if len(s):
            raise ValueError(
                f"{list(s)} not an element of ['R', 'F', 'U', 'RF', 'RU', 'FU', 'RFU']"
            )

        original_audio = (
            input.array if isinstance(original_audio, Frame) else original_audio
        )
        target_audio = (
            input.array if isinstance(target_audio, Frame) else target_audio
        )
        wav, mel, f0, v = self._get_data(original_audio)
        wav_1, mel_1, f0_1, v_1 = self._get_data(target_audio)
        mels, mel_lens = padding_sequence_nd(
            [mel, mel_1], dim=0, return_len=True
        )
        f0s, f0_lens = padding_sequence_nd(
            [f0, f0_1], dim=0, return_len=True
        )

        f0_org_quantized = quantize_f0_numpy(f0s[0, :, 0])[0]
        f0_org_onehot = f0_org_quantized[np.newaxis, :, :]
        uttr_f0_org = np.concatenate([mels[:1], f0_org_onehot], axis=-1)
        f0_trg_quantized = quantize_f0_numpy(f0s[1, :, 0])[0]
        f0_trg_onehot = f0_trg_quantized[np.newaxis, :, :]

        r = self._execute(
            inputs=[mels[:1], f0_trg_onehot, [len(f0s[0])]],
            input_labels=['X', 'f0_onehot', 'len_X'],
            output_labels=['f0_target'],
        )
        f0_pred = r['f0_target']
        f0_pred_quantized = f0_pred.argmax(axis=-1).squeeze(0)
        f0_con_onehot = np.zeros_like(f0_pred)
        f0_con_onehot[0, np.arange(f0_pred.shape[1]), f0_pred_quantized] = 1
        uttr_f0_trg = np.concatenate([mels[:1], f0_con_onehot], axis=-1)
        results = {}
        for condition in modes:
            if condition == 'R':
                uttr_f0_ = uttr_f0_org
                v_ = v
                x_ = mels[1:]
            if condition == 'F':
                uttr_f0_ = uttr_f0_trg
                v_ = v
                x_ = mels[:1]
            if condition == 'U':
                uttr_f0_ = uttr_f0_org
                v_ = v_1
                x_ = mels[:1]
            if condition == 'RF':
                uttr_f0_ = uttr_f0_trg
                v_ = v
                x_ = mels[1:]
            if condition == 'RU':
                uttr_f0_ = uttr_f0_org
                v_ = v_1
                x_ = mels[:1]
            if condition == 'FU':
                uttr_f0_ = uttr_f0_trg
                v_ = v_1
                x_ = mels[:1]
            if condition == 'RFU':
                uttr_f0_ = uttr_f0_trg
                v_ = v_1
                x_ = mels[:1]

            r = self._execute(
                inputs=[uttr_f0_, x_, [v_], [len(f0s[0])]],
                input_labels=['uttr_f0', 'X', 'V', 'len_X'],
                output_labels=['mel_outputs'],
            )
            mel_outputs = r['mel_outputs'][0]
            if 'R' in condition:
                length = mel_lens[1]
            else:
                length = mel_lens[0]
            mel_outputs = mel_outputs[:length]
            results[condition] = mel_outputs

        return results


class Fastpitch(Abstract):
    def __init__(
        self, input_nodes, output_nodes, normalizer, stats, sess, model, name
    ):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._normalizer = normalizer
        self._stats = stats
        self._sess = sess
        self.__model__ = model
        self.__name__ = name

    def predict(
        self,
        string,
        speed_ratio: float = 1.0,
        pitch_ratio: float = 1.0,
        pitch_addition: float = 0.0,
        **kwargs,
    ):
        """
        Change string to Mel.

        Parameters
        ----------
        string: str
        speed_ratio: float, optional (default=1.0)
            Increase this variable will increase time voice generated.
        pitch_ratio: float, optional (default=1.0)
            pitch = pitch * pitch_ratio, amplify existing pitch contour.
        pitch_addition: float, optional (default=0.0)
            pitch = pitch + pitch_addition, change pitch contour.

        Returns
        -------
        result: Dict[string, decoder-output, postnet-output, pitch-output, universal-output]
        """
        t, ids = self._normalizer.normalize(string, **kwargs)
        r = self._execute(
            inputs=[[ids], [speed_ratio], [pitch_ratio], [pitch_addition]],
            input_labels=[
                'Placeholder',
                'speed_ratios',
                'pitch_ratios',
                'pitch_addition',
            ],
            output_labels=['decoder_output', 'post_mel_outputs', 'pitch_outputs'],
        )
        v = r['post_mel_outputs'][0] * self._stats[1] + self._stats[0]
        v = (v - MEL_MEAN) / MEL_STD
        return {
            'string': t,
            'ids': ids,
            'decoder-output': r['decoder_output'][0],
            'postnet-output': r['post_mel_outputs'][0],
            'pitch-output': r['pitch_outputs'][0],
            'universal-output': v,
        }

    def __call__(self, input, **kwargs):
        return self.predict(input, **kwargs)


class TransducerAligner(Abstract):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        featurizer,
        vocab,
        time_reduction_factor,
        sess,
        model,
        name,
        wavs,
        dummy_sentences,
        stack=False,
    ):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._featurizer = featurizer
        self._vocab = vocab
        self._time_reduction_factor = time_reduction_factor
        self._sess = sess
        self.__model__ = model
        self.__name__ = name
        self._wavs = wavs
        self._dummy_sentences = dummy_sentences
        self._stack = stack
        if self._stack:
            self._len_vocab = [l.vocab_size for l in self._vocab]

    def _get_inputs(self, inputs, texts):
        inputs = [
            input.array if isinstance(input, Frame) else input
            for input in inputs
        ]

        index = len(inputs)

        # pretty hacky, result from single batch is not good caused by batchnorm.
        # have to append extra random wavs
        if len(inputs) < len(self._wavs) + 1:
            inputs = inputs + self._wavs[:(len(self._wavs) + 1) - len(inputs)]
            texts = texts + self._dummy_sentences

        padded, lens = sequence_1d(inputs, return_len=True)
        targets = [subword_encode(self._vocab, t) for t in texts]
        targets_padded, targets_lens = sequence_1d(targets, return_len=True)

        return padded, lens, targets_padded, targets_lens, index

    def _combined_indices(
        self, subwords, ids, l, reduction_factor=160, sample_rate=16000
    ):
        result, temp_l, temp_r = [], [], []
        for i in range(len(subwords)):
            if ids[i] is None and len(temp_r):
                data = {
                    'text': ''.join(temp_l),
                    'start': round(temp_r[0], 4),
                    'end': round(
                        temp_r[-1] + (reduction_factor / sample_rate), 4
                    ),
                }
                result.append(data)
                temp_l, temp_r = [], []
            else:
                temp_l.append(subwords[i])
                temp_r.append(l[ids[i]])

        if len(temp_l):
            data = {
                'text': ''.join(temp_l),
                'start': round(temp_r[0], 4),
                'end': round(temp_r[-1] + (reduction_factor / sample_rate), 4),
            }
            result.append(data)

        return result

    def predict(self, input, transcription: str):
        """
        Transcribe input, will return a string.

        Parameters
        ----------
        input: np.array
            np.array or malaya_speech.model.frame.Frame.
        transcription: str
            transcription of input audio

        Returns
        -------
        result: Dict[words_alignment, subwords_alignment, subwords, alignment]
        """

        padded, lens, targets_padded, targets_lens, index = self._get_inputs([input],
                                                                             [transcription])
        r = self._execute(
            inputs=[padded, lens, targets_padded, targets_lens],
            input_labels=['X_placeholder', 'X_len_placeholder', 'subwords', 'subwords_lens'],
            output_labels=['non_blank_transcript', 'non_blank_stime', 'decoded', 'alignment'],
        )
        non_blank_transcript = r['non_blank_transcript']
        non_blank_stime = r['non_blank_stime']
        decoded = r['decoded']
        alignment = r['alignment']
        if self._stack:
            words, indices = align_multilanguage(
                self._vocab, non_blank_transcript, get_index=True
            )
        else:
            words, indices = self._vocab.decode(
                non_blank_transcript, get_index=True
            )
        words_alignment = self._combined_indices(words, indices, non_blank_stime)

        words, indices = [], []
        for no, ids in enumerate(non_blank_transcript):
            if self._stack:
                last_index, v = get_index_multilanguage(ids, self._vocab, self._len_vocab)
                w = self._vocab[last_index]._id_to_subword(v - 1)
            else:
                w = self._vocab._id_to_subword(ids - 1)
            if isinstance(w, bytes):
                w = w.decode()
            words.extend([w, None])
            indices.extend([no, None])
        subwords_alignment = self._combined_indices(words, indices, non_blank_stime)

        if self._stack:
            subwords_ = []
            for ids in decoded[decoded > 0]:
                last_index, v = get_index_multilanguage(ids, self._vocab, self._len_vocab)
            subwords_.append(self._vocab[last_index]._id_to_subword(v - 1))
        else:
            subwords_ = [self._vocab._id_to_subword(ids - 1) for ids in decoded[decoded > 0]]
        subwords_ = [s.decode() if isinstance(s, bytes) else s for s in subwords_]
        alignment = alignment[:, targets_padded[0, :targets_lens[0]]].T
        return {'words_alignment': words_alignment,
                'subwords_alignment': subwords_alignment,
                'subwords': subwords_,
                'alignment': alignment}

    def __call__(self, input, transcription: str):
        """
        Transcribe input, will return a string.

        Parameters
        ----------
        input: np.array
            np.array or malaya_speech.model.frame.Frame.
        transcription: str
            transcription of input audio

        Returns
        -------
        result: Dict[words_alignment, subwords_alignment, subwords, alignment]
        """
        return self.predict(input, transcription)


class GlowTTS(Abstract):
    def __init__(
        self, input_nodes, output_nodes, normalizer, stats, sess, model, name, **kwargs
    ):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._normalizer = normalizer
        self._stats = stats
        self._sess = sess
        self.__model__ = model
        self.__name__ = name

    def predict(
        self,
        string,
        temperature: float = 0.3333,
        length_ratio: float = 1.0,
        **kwargs,
    ):
        """
        Change string to Mel.

        Parameters
        ----------
        string: str
        temperature: float, optional (default=0.3333)
            Decoder model trying to decode with encoder(text) + random.normal() * temperature.
        length_ratio: float, optional (default=1.0)
            Increase this variable will increase time voice generated.

        Returns
        -------
        result: Dict[string, ids, mel-output, alignment, universal-output]
        """
        t, ids = self._normalizer.normalize(string, **kwargs)
        r = self._execute(
            inputs=[[ids], [len(ids)], [temperature], [length_ratio]],
            input_labels=[
                'input_ids',
                'lens',
                'temperature',
                'length_ratio',
            ],
            output_labels=['mel_output', 'alignment_histories'],
        )
        v = r['mel_output'][0] * self._stats[1] + self._stats[0]
        v = (v - MEL_MEAN) / MEL_STD
        return {
            'string': t,
            'ids': ids,
            'mel-output': r['mel_output'][0],
            'alignment': r['alignment_histories'][0].T,
            'universal-output': v,
        }

    def __call__(self, input, **kwargs):
        return self.predict(input, **kwargs)


class GlowTTS_MultiSpeaker(Abstract):
    def __init__(
        self, input_nodes, output_nodes, normalizer, speaker_vector, stats, sess, model, name
    ):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._normalizer = normalizer
        self._speaker_vector = speaker_vector
        self._sess = sess
        self.__model__ = model
        self.__name__ = name

    def _predict(self, string, left_audio, right_audio,
                 temperature: float = 0.3333,
                 length_ratio: float = 1.0, **kwargs):
        t, ids = self._normalizer.normalize(string, **kwargs)
        left_v = self._speaker_vector([left_audio])
        right_v = self._speaker_vector([right_audio])
        r = self._execute(
            inputs=[[ids], [len(ids)], [temperature], [length_ratio], left_v, right_v],
            input_labels=[
                'input_ids',
                'lens',
                'temperature',
                'length_ratio',
                'speakers',
                'speakers_right',
            ],
            output_labels=['mel_output', 'alignment_histories'],
        )
        return {
            'string': t,
            'ids': ids,
            'alignment': r['alignment_histories'][0].T,
            'universal-output': r['mel_output'][0][:-8],
        }

    def predict(
        self,
        string,
        audio,
        temperature: float = 0.3333,
        length_ratio: float = 1.0,
        **kwargs,
    ):
        """
        Change string to Mel.

        Parameters
        ----------
        string: str
        audio: np.array
            np.array or malaya_speech.model.frame.Frame, must in 16k format.
            We only trained on `female`, `male`, `husein` and `haqkiem` speakers.
        temperature: float, optional (default=0.3333)
            Decoder model trying to decode with encoder(text) + random.normal() * temperature.
        length_ratio: float, optional (default=1.0)
            Increase this variable will increase time voice generated.

        Returns
        -------
        result: Dict[string, ids, alignment, universal-output]
        """
        return self._predict(string=string,
                             left_audio=audio, right_audio=audio,
                             temperature=temperature, length_ratio=length_ratio, **kwargs)

    def voice_conversion(self, string, original_audio, target_audio,
                         temperature: float = 0.3333,
                         length_ratio: float = 1.0,
                         **kwargs,):
        """
        Change string to Mel.

        Parameters
        ----------
        string: str
        original_audio: np.array
            original speaker to encode speaking style, must in 16k format.
        target_audio: np.array
            target speaker to follow speaking style from `original_audio`, must in 16k format.
        temperature: float, optional (default=0.3333)
            Decoder model trying to decode with encoder(text) + random.normal() * temperature.
        length_ratio: float, optional (default=1.0)
            Increase this variable will increase time voice generated.

        Returns
        -------
        result: Dict[string, ids, alignment, universal-output]
        """
        return self._predict(string=string,
                             left_audio=original_audio, right_audio=target_audio,
                             temperature=temperature, length_ratio=length_ratio, **kwargs)

    def __call__(self, input, **kwargs):
        return self.predict(input, **kwargs)
