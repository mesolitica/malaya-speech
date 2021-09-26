import re
from unidecode import unidecode
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np

_pad = 'pad'
_start = 'start'
_eos = 'eos'
_punctuation = "!'(),.:;? "
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_small_letters = 'abcdefghijklmnopqrstuvwxyz'
_rejected = '\'():;"'
_punct = ':;,.?'


TTS_SYMBOLS = (
    [_pad, _start, _eos] + list(_special) + list(_punctuation) + list(_letters)
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


class TextIDS:
    def __init__(
        self,
        pad_to: int = 8,
        understand_punct: bool = True,
        normalizer=None,
        sentence_tokenizer=None,
        true_case_model=None,
    ):
        self.normalizer = normalizer
        self.pad_to = pad_to
        self.sentence_tokenizer = sentence_tokenizer
        self.true_case_model = true_case_model
        self.understand_punct = understand_punct

    def normalize(
        self,
        string: str,
        normalize: bool = True,
        assume_newline_fullstop: bool = False,
        **kwargs
    ):
        """
        Normalize a string for TTS or force alignment task.

        Parameters
        ----------
        string: str
        normalize: bool, optional (default=True)
            will normalize the string using malaya.normalize.normalizer.
            will ignore this boolean if self.normalizer passed as None.
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

        string = string.replace('&', ' dan ')
        string = string.replace(':', ',').replace(';', ',')
        if normalize and self.normalizer is not None:
            t = self.normalizer._tokenizer(string)
            for i in range(len(t)):
                if t[i] == '-':
                    t[i] = ','
            string = ' '.join(t)
            string = self.normalizer.normalize(
                string,
                check_english=False,
                normalize_entity=False,
                normalize_text=False,
                normalize_url=True,
                normalize_email=True,
                normalize_telephone=True,
            )
            string = string['normalize']
        else:
            string = string
        string = put_spacing_num(string)
        string = ''.join(
            [
                c
                for c in string
                if c in TTS_SYMBOLS and c not in _rejected
            ]
        )
        if not self.understand_punct:
            string = ''.join([c for c in string if c not in _punct])
        string = re.sub(r'[ ]+', ' ', string).strip()
        string = string.lower()
        ids = tts_encode(string, TTS_SYMBOLS, add_eos=False)
        text_input = np.array(ids)
        num_pad = self.pad_to - ((len(text_input) + 2) % self.pad_to)
        text_input = np.pad(
            text_input, ((1, 1)), 'constant', constant_values=((1, 2))
        )
        text_input = np.pad(
            text_input, ((0, num_pad)), 'constant', constant_values=0
        )
        return string, text_input
