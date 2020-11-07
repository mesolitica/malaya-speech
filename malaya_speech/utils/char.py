import six
from typing import List

PAD = '<PAD>'
EOS = '<EOS>'
RESERVED_TOKENS = [PAD, EOS]
NUM_RESERVED_TOKENS = len(RESERVED_TOKENS)
PAD_ID = RESERVED_TOKENS.index(PAD)
EOS_ID = RESERVED_TOKENS.index(EOS)
VOCAB_SIZE = 256
BLANK = 0


def strip_ids(ids, ids_to_strip):
    """Strip ids_to_strip from the end ids."""
    ids = list(ids)
    while ids and ids[-1] in ids_to_strip:
        ids.pop()
    return ids


def generate_vocab(strings: List[str]):
    """
    Generate character vocab sorted based on frequency.

    Parameters
    -----------
    strings: List[str]

    Returns
    --------
    result: List[str]
    """
    joined = ' '.join(strings)
    unique_chars = set(joined)
    unique_chars = [(c, joined.count(c)) for c in unique_chars]
    unique_chars = sorted(
        unique_chars, key = lambda element: element[1], reverse = True
    )
    unique_chars, _ = zip(*unique_chars)
    unique_chars = list(unique_chars)
    return RESERVED_TOKENS + unique_chars


def encode(
    string: str,
    add_eos: bool = True,
    add_blank: bool = False,
    lookup: List[str] = None,
):
    """
    Encode string to integer representation based on ascii table or lookup variable.

    Parameters
    -----------
    string: str
    add_eos: bool, optional (default=True)
        add EOS token at the end of encoded.
    add_blank: bool, optional (default=False)
        add BLANK token at the starting of encoded, this is for transducer / transformer based.
    lookup: List[str], optional (default=None)
        list of unique strings.

    Returns
    --------
    result: List[int]
    """
    if lookup:
        if len(lookup) != len(set(lookup)):
            raise ValueError('lookup must be a list of unique strings')
        r = [lookup.index(c) for c in string]
    else:
        r = [c + NUM_RESERVED_TOKENS for c in string.encode('utf-8')]
    if add_eos:
        r = r + [1]
    if add_blank:
        r = [BLANK] + r
    return r


def decode(ids, lookup: List[str] = None):
    """
    Decode integer representation to string based on ascii table or lookup variable.

    Parameters
    -----------
    ids: List[int]
    lookup: List[str], optional (default=None)
        list of unique strings.

    Returns
    --------
    result: str
    """
    decoded_ids = []
    int2byte = six.int2byte
    for id_ in ids:
        if 0 <= id_ < NUM_RESERVED_TOKENS:
            decoded_ids.append(RESERVED_TOKENS[int(id_)])
        else:
            if lookup:
                decoded_ids.append(lookup[id_])
            else:
                decoded_ids.append(
                    int2byte(id_ - NUM_RESERVED_TOKENS).decode('utf-8')
                )

    return ''.join(decoded_ids)
