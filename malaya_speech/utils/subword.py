from typing import List
from malaya_speech.utils.text_encoder.subword_encoder import SubwordTextEncoder
import re

BLANK = 0


def generate_tokenizer(
    strings: List[str],
    target_vocab_size: int = 1024,
    max_subword_length: int = 4,
    max_corpus_chars=None,
    reserved_tokens=None,
):
    return SubwordTextEncoder.build_from_corpus(
        strings,
        target_vocab_size=target_vocab_size,
        max_subword_length=max_subword_length,
        max_corpus_chars=max_corpus_chars,
        reserved_tokens=reserved_tokens,
    )


def decode_multilanguage(row, langs):

    if not len(row):
        return ''

    len_vocab = [l.vocab_size for l in langs]

    def get_index_multilanguage(r):
        for i in range(len(langs)):
            sum_v = sum(len_vocab[:i + 1])
            if r < sum(len_vocab[:i + 1]):
                return i, r - sum(len_vocab[:i])

    last_index, v = get_index_multilanguage(row[0])
    d, q = [], [v]
    for r in row[1:]:
        index, v = get_index_multilanguage(r)
        if index != last_index:
            d.append(decode(langs[last_index], q))
            q = [v]
            last_index = index
        else:
            q.append(v)
    if len(q):
        d.append(decode(langs[last_index], q))
    d = re.sub(r'[ ]+', ' ', ' '.join(d)).strip()
    d = d.replace(' lah', 'lah')
    return d


def encode(tokenizer, string: str, add_blank: bool = False):
    """
    Encode string to integer representation based on ascii table or lookup variable.

    Parameters
    -----------
    string: str
    add_blank: bool, optional (default=False)
        add BLANK token at the starting of encoded, this is for transducer / transformer based.
    lookup: List[str], optional (default=None)
        list of unique strings.

    Returns
    --------
    result: List[int]
    """
    r = tokenizer.encode(string)
    if add_blank:
        r = [BLANK] + r
    return r


def decode(tokenizer, ids):
    """
    Decode integer representation to string based on tokenizer vocab.

    Parameters
    -----------
    ids: List[int]

    Returns
    --------
    result: str
    """
    return tokenizer.decode([i for i in ids if i > 0])


def save(tokenizer, path: str):
    tokenizer.save_to_file(path)


def load(path: str):
    return SubwordTextEncoder.load_from_file(path)
