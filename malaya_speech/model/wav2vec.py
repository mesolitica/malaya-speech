import tensorflow as tf
import numpy as np
from malaya_speech.model.frame import Frame
from malaya_speech.utils.astype import int_to_float
from malaya_speech.utils.padding import sequence_1d
from malaya_speech.utils.char import CTC_VOCAB
from malaya_speech.utils.char import decode as char_decode
from malaya_speech.utils.activation import softmax
from malaya_speech.utils.read import resample
from malaya_speech.utils.aligner import (
    get_trellis,
    backtrack,
    merge_repeats,
    merge_words,
)
from malaya_speech.model.abstract import Abstract
from scipy.special import log_softmax
from typing import Callable


class CTC(Abstract):
    def __init__(self, input_nodes, output_nodes, sess, model, name):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._sess = sess
        self.__model__ = model
        self.__name__ = name


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

    def beam_decoder(self, inputs, beam_width: int = 100, **kwargs):
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

    def predict_logits(self, inputs, norm_func=softmax):
        """
        Predict logits from inputs.

        Parameters
        ----------
        input: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].
        norm_func: Callable, optional (default=malaya.utils.activation.softmax)


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
        logits = norm_func(logits, axis=-1)
        results = []
        for i in range(len(logits)):
            results.append(logits[i][: seq_lens[i]])
        return results

    def gradio(self, record_mode: bool = True,
               lm_func: Callable = None,
               **kwargs):
        """
        Transcribe an input using beam decoder on Gradio interface.

        Parameters
        ----------
        record_mode: bool, optional (default=True)
            if True, Gradio will use record mode, else, file upload mode.
        lm_func: Callable, optional (default=None)
            if not None, will pass a logits with shape [T, D].

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
            if lm_func is not None:
                logits = self.predict_logits(inputs=[data])[0]
                return lm_func(logits)
            else:
                return self.beam_decoder(inputs=[data], **kwargs)[0]

        title = 'Wav2Vec2-STT using Beam Decoder'
        if lm_func is not None:
            title = f'{title} with LM'

        description = 'It will take sometime for the first time, after that, should be really fast.'

        if record_mode:
            input = 'microphone'
        else:
            input = 'audio'

        iface = gr.Interface(pred, input, 'text', title=title, description=description)
        return iface.launch(**kwargs)

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


class Wav2Vec2_Aligner(Abstract):
    def __init__(self, input_nodes, output_nodes, sess, model, name):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._sess = sess
        self.__model__ = model
        self.__name__ = name

    def _get_logits(self, padded, lens):
        r = self._execute(
            inputs=[padded, lens],
            input_labels=['X_placeholder', 'X_len_placeholder'],
            output_labels=['logits', 'seq_lens'],
        )
        return r['logits'], r['seq_lens']

    def predict(self, input, transcription: str, sample_rate: int = 16000):
        """
        Transcribe input, will return a string.

        Parameters
        ----------
        input: np.array
            np.array or malaya_speech.model.frame.Frame.
        transcription: str
            transcription of input audio.
        sample_rate: int, optional (default=16000)
            sample rate for `input`.
        Returns
        -------
        result: Dict[chars_alignment, words_alignment, alignment]
        """

        logits, seq_lens = self._get_logits([input], [len(input)])
        logits = np.transpose(logits, axes=(1, 0, 2))
        o = log_softmax(logits, axis=-1)[0]
        dictionary = {c: i for i, c in enumerate(CTC_VOCAB)}
        tokens = [dictionary[c] for c in transcription]
        trellis = get_trellis(o, tokens)
        path = backtrack(trellis, o, tokens)
        segments = merge_repeats(path, transcription)
        word_segments = merge_words(segments)

        t = (len(input) / sample_rate) / o.shape[0]
        chars_alignment = []
        for s in segments:
            chars_alignment.append({'text': s.label,
                                    'start': s.start * t,
                                    'end': s.end * t,
                                    'start_t': s.start,
                                    'end_t': s.end,
                                    'score': s.score})

        words_alignment = []
        for s in word_segments:
            words_alignment.append({'text': s.label,
                                    'start': s.start * t,
                                    'end': s.end * t,
                                    'start_t': s.start,
                                    'end_t': s.end,
                                    'score': s.score})

        return {
            'chars_alignment': chars_alignment,
            'words_alignment': words_alignment,
            'alignment': trellis,
        }

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
        result: Dict[chars_alignment, words_alignment, alignment]
        """
        return self.predict(input, transcription)
