import tensorflow as tf
import numpy as np
from itertools import groupby
from malaya_speech.model.frame import Frame
from malaya_speech.utils.astype import int_to_float
from malaya_speech.utils.padding import sequence_1d
from malaya_speech.utils.char import HF_CTC_VOCAB, HF_CTC_VOCAB_IDX
from malaya_speech.utils.char import decode as char_decode
from malaya_speech.utils.read import resample
from malaya_speech.utils.activation import softmax
from malaya_speech.utils.aligner import (
    get_trellis,
    backtrack,
    merge_repeats,
    merge_words,
)
from malaya_speech.model.abstract import Abstract
from scipy.special import log_softmax
from typing import Callable


def batching(audios):
    batch, lens = sequence_1d(audios, return_len=True)
    attentions = [[1] * l for l in lens]
    attentions = sequence_1d(attentions)
    normed_input_values = []

    for vector, length in zip(batch, attentions.sum(-1)):
        normed_slice = (vector - vector[:length].mean()) / np.sqrt(vector[:length].var() + 1e-7)
        if length < normed_slice.shape[0]:
            normed_slice[length:] = 0.0

        normed_input_values.append(normed_slice)

    normed_input_values = np.array(normed_input_values)
    return normed_input_values.astype(np.float32), attentions


class HuggingFace_CTC(Abstract):
    def __init__(self, hf_model, model, name):
        self.hf_model = hf_model
        self.__model__ = model
        self.__name__ = name

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
        logits = self.predict_logits(inputs=inputs)
        argmax = np.argmax(logits, axis=-1)

        results = []
        for i in range(len(argmax)):
            tokens = char_decode(argmax[i], lookup=HF_CTC_VOCAB + ['_'])
            grouped_tokens = [token_group[0] for token_group in groupby(tokens)]
            filtered_tokens = list(filter(lambda token: token != '_', grouped_tokens))
            r = ''.join(filtered_tokens).strip()
            results.append(r)
        return results

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
        normed_input_values, attentions = batching(inputs)
        out = self.hf_model(normed_input_values, attention_mask=attentions)
        return norm_func(out[0].numpy(), axis=-1)

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

        **kwargs: keyword arguments for `iface.launch`.
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
                return self.greedy_decoder(inputs=[data])[0]

        title = 'HuggingFace-Wav2Vec2-STT'
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


class HuggingFace_Aligner(Abstract):
    def __init__(self, hf_model, model, name):
        self.hf_model = hf_model
        self.__model__ = model
        self.__name__ = name

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

        input = input.array if isinstance(input, Frame) else input
        normed_input_values, attentions = batching([input])
        out = self.hf_model(normed_input_values, attention_mask=attentions)
        logits = out[0].numpy()
        o = log_softmax(logits, axis=-1)[0]
        tokens = [HF_CTC_VOCAB_IDX[c] for c in transcription]
        trellis = get_trellis(o, tokens, blank_id=len(HF_CTC_VOCAB))
        path = backtrack(trellis, o, tokens, blank_id=len(HF_CTC_VOCAB))
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
