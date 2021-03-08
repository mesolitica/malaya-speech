from typing import List
from malaya_speech.utils.text_encoder.subword_encoder import SubwordTextEncoder

BLANK = 0


def generate_tokenizer(
    strings: List[str],
    target_vocab_size: int = 1024,
    max_subword_length: int = 4,
    max_corpus_chars = None,
    reserved_tokens = None,
):
    return SubwordTextEncoder.build_from_corpus(
        strings,
        target_vocab_size = target_vocab_size,
        max_subword_length = max_subword_length,
        max_corpus_chars = max_corpus_chars,
        reserved_tokens = reserved_tokens,
    )


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
