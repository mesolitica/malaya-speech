import re
from unidecode import unidecode
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np
import inspect
from typing import Callable

_pad = 'pad'
_start = 'start'
_eos = 'eos'
_punctuation = "!'(),.:;? "
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_numbers = '0123456789'
_small_letters = 'abcdefghijklmnopqrstuvwxyz'
_rejected = '\'():;"'
_punct = ':;,.?'
_bracket = '()[]'

PRONUNCIATION = {
    'A': 'ae',
    'B': 'bi',
    'C': 'si',
    'D': 'di',
    'E': 'ei',
    'F': 'ef',
    'G': 'ji',
    'H': 'hesh',
    'I': 'ai',
    'J': 'jei',
    'K': 'kei',
    'L': 'el',
    'M': 'eim',
    'N': 'ein',
    'O': 'ou',
    'P': 'pi',
    'Q': 'qeu',
    'R': 'ar',
    'S': 'es',
    'T': 'ti',
    'U': 'yu',
    'V': 'vi',
    'W': 'dablui',
    'X': 'ex',
    'Y': 'wai',
    'Z': 'zed',
}

TTS_SYMBOLS = (
    [_pad, _start, _eos] + list(_special) + list(_punctuation) + list(_letters)
)

TTS_AZURE_SYMBOLS = (
    [_pad, _start, _eos] + list(_special) + list(_punctuation) + list(_letters) + list(_numbers)
)

FORCE_ALIGNMENT_SYMBOLS = (
    [_pad, _start, _eos] + list(_special) + list(_small_letters)
)


def convert_to_ascii(string):
    return unidecode(string)


def collapse_whitespace(string):
    return re.sub(_whitespace_re, ' ', string)


def put_spacing_num(string):
    """
    'ni1996' -> 'ni 1996'
    """
    string = re.sub('[A-Za-z]+', lambda ele: ' ' + ele[0] + ' ', string)
    return re.sub(r'[ ]+', ' ', string).strip()


def compute_sparse_correlation_matrix(A):
    scaler = StandardScaler(with_mean=False)
    scaled_A = scaler.fit_transform(A)
    corr_matrix = (1 / scaled_A.shape[0]) * (scaled_A.T @ scaled_A)
    return corr_matrix


def filter_splitted(s, t, threshold=0.001):
    bow = CountVectorizer(token_pattern='[A-Za-z0-9]+').fit([s])
    s_bow = bow.transform([s] + t)
    score = np.array(compute_sparse_correlation_matrix(s_bow.T).todense())[
        0, 1:
    ]
    r = [t[no] for no, s in enumerate(score) if s >= threshold]
    return ' '.join(r)


def tts_encode(string: str, vocab, add_eos: bool = True):
    r = [vocab.index(c) for c in string]
    if add_eos:
        r = r + [vocab.index('eos')]
    return r


class TextIDS_AZURE:
    def __init__(
        self,
        pad_to: int = 8,
        sentence_tokenizer=None,
        true_case_model=None,
    ):
        self.pad_to = pad_to
        self.sentence_tokenizer = sentence_tokenizer
        self.true_case_model = true_case_model

    def normalize(
        self,
        string: str,
        assume_newline_fullstop: bool = False,
        **kwargs
    ):
        """
        Normalize a string for TTS task using Azure dataset mode.

        Parameters
        ----------
        string: str
        assume_newline_fullstop: bool, optional (default=False)
            Assume a string is a multiple sentences, will split using
            `malaya.text.function.split_into_sentences`.

        Returns
        -------
        result : (string: str, text_input: np.array)
        """
        string = convert_to_ascii(string)
        if assume_newline_fullstop and self.sentence_tokenizer is not None:
            string = string.replace('\n', '. ')
            string = self.sentence_tokenizer(string, minimum_length=0)
            string = '. '.join(string)

        if self.true_case_model is not None:
            string = self.true_case_model(string)

        string = re.sub(r'[ ]+', ' ', string).strip()
        if string[-1] in '-,':
            string = string[:-1]
        if string[-1] not in '.,?!':
            string = string + '.'

        string = put_spacing_num(string)
        string = ''.join([c for c in string if c in TTS_AZURE_SYMBOLS])
        string = re.sub(r'[ ]+', ' ', string).strip()
        ids = tts_encode(string, TTS_AZURE_SYMBOLS, add_eos=False)
        text_input = np.array(ids)
        num_pad = self.pad_to - ((len(text_input) + 2) % self.pad_to)
        text_input = np.pad(
            text_input, ((1, 1)), 'constant', constant_values=((1, 2))
        )
        text_input = np.pad(
            text_input, ((0, num_pad)), 'constant', constant_values=0
        )
        return string, text_input


class TextIDS:
    def __init__(
        self,
        pad_to: int = 8,
        understand_punct: bool = True,
        is_lower: bool = True,
        normalizer=None,
        sentence_tokenizer=None,
    ):
        self.pad_to = pad_to
        self.understand_punct = understand_punct
        self.is_lower = is_lower
        self.normalizer = normalizer
        self.sentence_tokenizer = sentence_tokenizer

        if self.normalizer is not None:
            normalizer_parameters = list(inspect.signature(self.normalizer.normalize).parameters)
            self.kwargs = {
                'normalize_entity': False,
                'normalize_text': False,
                'normalize_url': True,
                'normalize_email': True,
                'normalize_telephone': True,
            }
            if 'check_english_func' in normalizer_parameters:
                self.kwargs['check_english_func'] = None
            else:
                self.kwargs['check_english'] = False
        else:
            self.kwargs = {}

    def normalize(
        self,
        string: str,
        normalize: bool = True,
        replace_brackets_with_comma: bool = True,
        assume_newline_fullstop: bool = False,
        true_case_func: Callable = None,
        add_fullstop: bool = True,
        **kwargs,
    ):
        """
        Normalize a string for TTS task.

        Parameters
        ----------
        string: str
        normalize: bool, optional (default=True)
            will normalize the string using malaya.normalize.normalizer.
            will ignore this boolean if self.normalizer passed as None.
        replace_brackets_with_comma: bool, optional (default=True)
            will replace [text] / (text) -> , text ,
        assume_newline_fullstop: bool, optional (default=False)
            Assume a string is a multiple sentences, will split using
            `malaya.text.function.split_into_sentences`.
        true_case_func: Callable, optional (default=None)
            Callable function to do true case, eg, https://malaya.readthedocs.io/en/latest/load-true-case.html
            Only useful for TTS models that understood uppercase.

        Returns
        -------
        result : (string: str, text_input: np.array)
        """

        string = convert_to_ascii(string)
        if assume_newline_fullstop and self.sentence_tokenizer is not None:
            string = string.replace('\n', '. ')
            string = self.sentence_tokenizer(string, minimum_length=0)
            string = '. '.join(string)

        string = re.sub(r'[ ]+', ' ', string).strip()
        if string[-1] in '-,':
            string = string[:-1]
        if add_fullstop and string[-1] not in '.,?!':
            string = string + '.'

        string = string.replace(':', ',').replace(';', ',')
        if normalize and self.normalizer is not None:
            string = self.normalizer.normalize(string, **{**self.kwargs, **kwargs})
            string = string['normalize']
        else:
            string = string
        string = put_spacing_num(string)
        string = ''.join([c for c in string if c in TTS_SYMBOLS])

        if true_case_func is not None:
            string = true_case_func(string)

        if self.is_lower:
            string = string.lower()

        if replace_brackets_with_comma:
            string = ''.join([' , ' if c in _bracket else c for c in string if c])

        if not self.understand_punct:
            string = ''.join([c for c in string if c not in _punct])

        string = re.sub(r'(, )+', ', ', string)
        string = re.sub(r'[ ]+', ' ', string).strip()

        if add_fullstop and string[-1] not in '.,?!':
            string = string + ' .'

        ids = tts_encode(string, TTS_SYMBOLS, add_eos=False)
        text_input = np.array(ids)
        if self.pad_to is not None:
            num_pad = self.pad_to - ((len(text_input) + 2) % self.pad_to)
            text_input = np.pad(
                text_input, ((1, 1)), 'constant', constant_values=((1, 2))
            )
            text_input = np.pad(
                text_input, ((0, num_pad)), 'constant', constant_values=0
            )
        return string, text_input
