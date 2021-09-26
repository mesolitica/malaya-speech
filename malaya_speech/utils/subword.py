from .text_encoder.subword_encoder import SubwordTextEncoder, _trim_underscore_and_tell
from .text_encoder import pad_decr
import re
import six
from typing import List

BLANK = 0


def get_index_multilanguage(r, tokenizers, len_vocab):
    for i in range(len(tokenizers)):
        sum_v = sum(len_vocab[:i + 1])
        if r < sum(len_vocab[:i + 1]):
            return i, r - sum(len_vocab[:i])


def generate_tokenizer(
    strings: List[str],
    target_vocab_size: int = 1024,
    max_subword_length: int = 4,
    max_corpus_chars=None,
    reserved_tokens=None,
):
    """
    Build a subword dictionary.
    """
    return SubwordTextEncoder.build_from_corpus(
        strings,
        target_vocab_size=target_vocab_size,
        max_subword_length=max_subword_length,
        max_corpus_chars=max_corpus_chars,
        reserved_tokens=reserved_tokens,
    )


def save(tokenizer, path: str):
    """
    Save subword dictionary to a text file.
    """
    tokenizer.save_to_file(path)


def load(path: str):
    """
    Load text file into subword dictionary.
    """
    return SubwordTextEncoder.load_from_file(path)


def encode(tokenizer, string: str, add_blank: bool = False):
    """
    Encode string to integer representation based on ascii table or lookup variable.

    Parameters
    -----------
    tokenizer: object
        tokenizer object
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
    tokenizer: object
        tokenizer object
    ids: List[int]

    Returns
    --------
    result: str
    """
    return tokenizer.decode([i for i in ids if i > 0])


def decode_multilanguage(tokenizers, ids):
    """
    Decode integer representation to string using list of tokenizer objects.

    Parameters
    -----------
    tokenizers: List[object]
        List of tokenizer objects.
    ids: List[int]

    Returns
    --------
    result: str
    """

    if not len(ids):
        return ''

    len_vocab = [l.vocab_size for l in tokenizers]

    last_index, v = get_index_multilanguage(ids[0], tokenizers, len_vocab)
    d, q = [], [v]
    for r in ids[1:]:
        index, v = get_index_multilanguage(r, tokenizers, len_vocab)
        if index != last_index:
            d.append(decode(tokenizers[last_index], q))
            q = [v]
            last_index = index
        else:
            q.append(v)
    if len(q):
        d.append(decode(tokenizers[last_index], q))
    d = re.sub(r'[ ]+', ' ', ' '.join(d)).strip()
    return d


def align_multilanguage(tokenizers, ids, get_index=False):
    ids = pad_decr(ids)
    subword_ids = ids

    subwords_ = []
    prev_bytes = []
    prev_ids = []
    ids = []

    len_vocab = [l.vocab_size for l in tokenizers]

    def consume_prev_bytes():
        if prev_bytes:
            subwords_.extend(prev_bytes)
            ids.extend(prev_ids)
        return [], []

    for no, subword_id in enumerate(subword_ids):
        last_index, v = get_index_multilanguage(subword_id, tokenizers, len_vocab)
        subword = tokenizers[last_index]._id_to_subword(v)
        if isinstance(subword, six.binary_type):
            # Byte-encoded
            prev_bytes.append(subword.decode('utf-8', 'replace'))
            if subword == b' ':
                prev_ids.append(None)
            else:
                prev_ids.append(no)
        else:
            # If there were bytes previously, convert to unicode.
            prev_bytes, prev_ids = consume_prev_bytes()
            trimmed, add_space = _trim_underscore_and_tell(subword)
            ids.append(no)
            subwords_.append(trimmed)
            if add_space:
                subwords_.append(' ')
                ids.append(None)
    prev_bytes = consume_prev_bytes()

    if get_index:
        return subwords_, ids
    else:
        return tf.compat.as_text(''.join(subwords_))

    len_vocab = [l.vocab_size for l in tokenizers]
