import torch
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
from malaya_speech.utils.subword import merge_bpe_tokens
from malaya_speech.model.abstract import Abstract
from malaya_boilerplate.torch_utils import to_tensor_cuda, to_numpy
from scipy.special import log_softmax
from typing import Callable
import logging

logger = logging.getLogger(__name__)

whisper_available = False
try:
    import whisper
    whisper_available = True
except Exception as e:
    logger.warning(
        '`openai-whisper` is not available, native whisper processor is not available, will use huggingface processor instead.')


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


class CTC(torch.nn.Module):
    def __init__(self, hf_model, model, name):
        super().__init__()
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
        cuda = next(self.hf_model.parameters()).is_cuda
        normed_input_values, attentions = batching(inputs)
        normed_input_values = to_tensor_cuda(torch.tensor(normed_input_values), cuda)
        attentions = to_tensor_cuda(torch.tensor(attentions), cuda)
        out = self.hf_model(normed_input_values, attention_mask=attentions)
        return norm_func(to_numpy(out[0]), axis=-1)

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


class Aligner(torch.nn.Module):
    def __init__(self, hf_model, model, name):
        super().__init__()
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
        cuda = next(self.hf_model.parameters()).is_cuda
        normed_input_values, attentions = batching([input])
        normed_input_values = to_tensor_cuda(torch.tensor(normed_input_values), cuda)
        attentions = to_tensor_cuda(torch.tensor(attentions), cuda)
        out = self.hf_model(normed_input_values, attention_mask=attentions)
        logits = to_numpy(out[0])
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


class Seq2Seq(torch.nn.Module):
    def __init__(self, hf_model, processor, model, name, use_whisper_processor=False, **kwargs):
        super().__init__()
        self.hf_model = hf_model
        self.processor = processor
        self.__model__ = model
        self.__name__ = name

        if use_whisper_processor:
            if 'whisper' not in model.lower():
                logger.warning(
                    '`use_whisper_processor` only available for whisper model, will fallback to huggingface processor')
                use_whisper_processor = False

            if not whisper_available:
                logger.warning(
                    'openai-whisper not installed. Please install it by `pip install openai-whisper` and try again. Will fallback to huggingface processor')
                use_whisper_processor = False

        self.use_whisper_processor = use_whisper_processor

    def generate(self, inputs, skip_special_tokens: bool = True, **kwargs):
        """
        Transcribe inputs.

        Returns
        -------
        result: List[str]

        Parameters
        ----------
        input: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].
        skip_special_tokens: bool, optional (default=True)
            skip special tokens during decoding.
        **kwargs: vector arguments pass to huggingface `generate` method.
            Read more at https://huggingface.co/docs/transformers/main_classes/text_generation

        Returns
        -------
        result: List[str]
        """
        inputs = [
            input.array if isinstance(input, Frame) else input
            for input in inputs
        ]
        cuda = next(self.hf_model.parameters()).is_cuda

        if self.use_whisper_processor:

            mels = []
            for k in range(len(inputs)):
                audio = whisper.pad_or_trim(inputs[k].astype(np.float32))
                mel = whisper.log_mel_spectrogram(audio)
                mels.append({'input_features': mel})

            batch = self.processor.feature_extractor.pad(mels, return_tensors="pt")
            input_features = batch.input_features

        else:
            input_features = self.processor(
                inputs, return_tensors='pt', sampling_rate=16000).input_features

        input_features = to_tensor_cuda(input_features, cuda)
        outputs = self.hf_model.generate(input_features, **kwargs)
        return self.processor.tokenizer.batch_decode(
            outputs, skip_special_tokens=skip_special_tokens)

    def predict_logits(self, inputs, norm_func=softmax, **kwargs):
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

        if kwargs.get('num_beams', 0) > 0:
            raise ValueError('beam decoding is not supported.')

        outputs = self.generate(
            inputs=inputs,
            output_attentions=True,
            output_hidden_states=True,
            output_scores=True,
            return_dict_in_generate=True,
            **kwargs,
        )
        stacked = torch.stack(outputs.scores)
        return to_numpy(stacked)

    def __call__(self, input, **kwargs):
        """
        Transcribe input.

        Parameters
        ----------
        input: np.array
            np.array or malaya_speech.model.frame.Frame.

        Returns
        -------
        result: str
        """
        return self.generate([input], **kwargs)[0]


class Seq2SeqAligner(torch.nn.Module):
    def __init__(self, hf_model, processor, model, name, **kwargs):
        super().__init__()

        self.hf_model = hf_model
        self.processor = processor
        self.tokenizer = self.processor.tokenizer
        self.__model__ = model
        self.__name__ = name

        self.AUDIO_SAMPLES_PER_TOKEN = processor.feature_extractor.hop_length * 2
        self.AUDIO_TIME_PER_TOKEN = self.AUDIO_SAMPLES_PER_TOKEN / processor.feature_extractor.sampling_rate

    def predict(
        self,
        input,
        transcription: str,
        lang: str = 'ms',
        median_filter_size: int = 7,
    ):
        """
        Transcribe input, will return a string.
        Based on https://github.com/openai/whisper/blob/main/notebooks/Multilingual_ASR.ipynb

        Parameters
        ----------
        input: np.array
            np.array or malaya_speech.model.frame.Frame.
        transcription: str
            transcription of input audio.
        lang: str, optional (default='ms')
            if you feed singlish speech, it is better to give `en` language.
        median_filter_size: int, optional (default=7)
            sliding median size.
        Returns
        -------
        result: Dict[chars_alignment, words_alignment, alignment]
        """

        try:
            from dtw import dtw
            from scipy.signal import medfilt
        except Exception as e:
            raise ModuleNotFoundError(
                'dtw-python not installed. Please install it by `pip install dtw-python` and try again.'
            )

        input = input.array if isinstance(input, Frame) else input
        cuda = next(self.hf_model.parameters()).is_cuda

        input_features = self.processor([input], return_tensors='pt').input_features
        input_features = to_tensor_cuda(input_features, cuda)

        label = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(
            f'<|startoftranscript|><|{lang}|><|transcribe|><|notimestamps|>{transcription}<|endoftext|>'))
        labels = self.tokenizer.pad([{'input_ids': label}], return_tensors='pt')

        with torch.no_grad():
            o = self.hf_model(
                input_features=input_features,
                labels=labels['input_ids'],
                output_attentions=True,
                return_dict=True,
            )

        duration = len(input)

        weights = torch.cat(o['cross_attentions'])
        weights = weights[:, :, :, : duration // self.AUDIO_SAMPLES_PER_TOKEN].cpu()
        weights = medfilt(weights, (1, 1, 1, median_filter_size))
        weights = torch.tensor(weights).softmax(dim=-1)
        w = weights / weights.norm(dim=-2, keepdim=True)
        matrix = w.mean(axis=(0, 1))
        alignment = dtw(-matrix.double().numpy())

        xticks = np.arange(0, matrix.shape[1], 1 / self.AUDIO_TIME_PER_TOKEN)
        xticklabels = (xticks * self.AUDIO_TIME_PER_TOKEN).round().astype(np.int32)

        yticklabels = self.tokenizer.convert_ids_to_tokens(labels['input_ids'][0])
        yticks = np.arange(len(yticklabels))

        jumps = np.pad(np.diff(alignment.index1s), (1, 0), constant_values=1).astype(bool)
        jump_times = alignment.index2s[jumps] * self.AUDIO_TIME_PER_TOKEN

        subwords_alignment = []
        for i in range(len(yticklabels)):
            d = {
                'text': yticklabels[i],
                'start': 0.0 if i == 0 else jump_times[i - 1],
                'end': jump_times[i]
            }
            subwords_alignment.append(d)

        merged_bpes = merge_bpe_tokens(
            zip(yticklabels, subwords_alignment), rejected=self.tokenizer.all_special_tokens)
        words_alignment = []
        for m in merged_bpes:
            if isinstance(m[1], list):
                start = m[1][0]['start']
                end = m[1][-1]['end']
            else:
                start = m[1]['start']
                end = m[1]['end']
            words_alignment.append({
                'text': m[0],
                'start': start,
                'end': end,
            })

        alignment_x = alignment.index2s
        alignment_y = alignment.index1s
        return {
            'subwords_alignment': subwords_alignment,
            'words_alignment': words_alignment,
            'alignment': to_numpy(matrix),
            'alignment_x': alignment_x,
            'alignment_y': alignment_y,
            'xticks': xticks,
            'xticklabels': xticklabels,
            'yticks': yticks,
            'yticklabels': yticklabels,
        }


class XVector(torch.nn.Module):
    def __init__(self, hf_model, processor, model, name):
        super().__init__()

        self.hf_model = hf_model
        self.processor = processor
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
        cuda = next(self.hf_model.parameters()).is_cuda

        inputs = self.processor(inputs, return_tensors='pt', sampling_rate=16000, padding=True)
        for k in inputs.keys():
            inputs[k] = to_tensor_cuda(inputs[k], cuda)

        embeddings = self.hf_model(**inputs).embeddings
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return to_numpy(embeddings)

    def forward(self, inputs):
        return self.vectorize(inputs)
