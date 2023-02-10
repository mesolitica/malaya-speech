import numpy as np
from malaya_speech.model.frame import Frame
from malaya_speech.utils.astype import int_to_float
from malaya_speech.utils.padding import sequence_1d
from malaya_speech.utils.subword import (
    decode as subword_decode,
    encode as subword_encode,
    decode_multilanguage,
    get_index_multilanguage,
    align_multilanguage,
)
from malaya_speech.utils.activation import apply_temp
from malaya_speech.utils.read import resample
from malaya_speech.utils.lm import (
    BeamHypothesis_LM,
    BeamHypothesis,
    sort_and_trim_beams,
    prune_history,
    get_lm_beams,
)
from malaya_speech.model.abstract import Abstract


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
                    'start': temp_r[0],
                    'end': temp_r[-1] + (reduction_factor / sample_rate),
                }
                result.append(data)
                temp_l, temp_r = [], []
            else:
                temp_l.append(subwords[i])
                temp_r.append(l[ids[i]])

        if len(temp_l):
            data = {
                'text': ''.join(temp_l),
                'start': temp_r[0],
                'end': temp_r[-1] + (reduction_factor / sample_rate),
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
                    w = self._vocabs.get(k, ' ')
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

    def gradio(self, record_mode: bool = True, **kwargs):
        """
        Transcribe an input using beam decoder on Gradio interface.

        Parameters
        ----------
        record_mode: bool, optional (default=True)
            if True, Gradio will use record mode, else, file upload mode.

        **kwargs: keyword arguments for beam decoder and `iface.launch`.
        """
        try:
            import gradio as gr
        except BaseException:
            raise ModuleNotFoundError(
                'gradio not installed. Please install it by `pip install gradio` and try again.'
            )

        def pred(audio):
            sample_rate, data = audio
            if len(data.shape) == 2:
                data = np.mean(data, axis=1)
            data = int_to_float(data)
            data = resample(data, sample_rate, 16000)
            return self._beam(inputs=[data], **kwargs)[0]

        title = 'RNNT-STT using Beam Decoder'
        description = 'It will take sometime for the first time, after that, should be really fast.'

        if record_mode:
            input = 'microphone'
        else:
            input = 'audio'

        iface = gr.Interface(pred, input, 'text', title=title, description=description)
        return iface.launch(**kwargs)

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
        return self.predict([input])[0]


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
                    'start': temp_r[0],
                    'end': temp_r[-1] + (reduction_factor / sample_rate),
                }
                result.append(data)
                temp_l, temp_r = [], []
            else:
                temp_l.append(subwords[i])
                temp_r.append(l[ids[i]])

        if len(temp_l):
            data = {
                'text': ''.join(temp_l),
                'start': temp_r[0],
                'end': temp_r[-1] + (reduction_factor / sample_rate),
            }
            result.append(data)

        return result

    def predict(self, input, transcription: str, sample_rate: int = 16000):
        """
        Transcribe input, will return a string.

        Parameters
        ----------
        input: np.array
            np.array or malaya_speech.model.frame.Frame.
        transcription: str
            transcription of input audio
        sample_rate: int, optional (default=16000)
            sample rate for `input`.

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
        alignment = alignment[:, targets_padded[0, :targets_lens[0]]]

        t = (len(input) / sample_rate) / alignment.shape[0]

        for i in range(len(words_alignment)):
            start = int(round(words_alignment[i]['start'] / t))
            end = int(round(words_alignment[i]['end'] / t))
            words_alignment[i]['start_t'] = start
            words_alignment[i]['end_t'] = end
            words_alignment[i]['score'] = alignment[start: end + 1, i].max()

        for i in range(len(subwords_alignment)):
            start = int(round(subwords_alignment[i]['start'] / t))
            end = int(round(subwords_alignment[i]['end'] / t))
            subwords_alignment[i]['start_t'] = start
            subwords_alignment[i]['end_t'] = end
            subwords_alignment[i]['score'] = alignment[start: end + 1, i].max()

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
