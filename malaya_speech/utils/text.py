import re
from unidecode import unidecode
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np


def convert_to_ascii(string):
    return unidecode(string)


def collapse_whitespace(string):
    return re.sub(_whitespace_re, ' ', string)


def put_spacing_num(string):
    string = re.sub('[A-Za-z]+', lambda ele: ' ' + ele[0] + ' ', string)
    return re.sub(r'[ ]+', ' ', string).strip()


def compute_sparse_correlation_matrix(A):
    scaler = StandardScaler(with_mean = False)
    scaled_A = scaler.fit_transform(A)
    corr_matrix = (1 / scaled_A.shape[0]) * (scaled_A.T @ scaled_A)
    return corr_matrix


def filter_splitted(s, t, threshold = 0.001):
    bow = CountVectorizer(token_pattern = '[A-Za-z0-9]+').fit([s])
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
