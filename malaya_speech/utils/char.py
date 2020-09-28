import six

PAD = '<pad>'
EOS = '<EOS>'
RESERVED_TOKENS = [PAD, EOS]
NUM_RESERVED_TOKENS = len(RESERVED_TOKENS)
PAD_ID = RESERVED_TOKENS.index(PAD)
EOS_ID = RESERVED_TOKENS.index(EOS)
VOCAB_SIZE = 256

if six.PY2:
    RESERVED_TOKENS_BYTES = RESERVED_TOKENS
else:
    RESERVED_TOKENS_BYTES = [bytes(PAD, 'ascii'), bytes(EOS, 'ascii')]


def strip_ids(ids, ids_to_strip):
    """Strip ids_to_strip from the end ids."""
    ids = list(ids)
    while ids and ids[-1] in ids_to_strip:
        ids.pop()
    return ids


def encode(string, add_eos = True):
    """
    Encode string to integer representation based on ascii table.

    Parameters
    -----------
    string: str

    Returns
    --------
    result: List[int]
    """
    numres = NUM_RESERVED_TOKENS
    if six.PY2:
        if isinstance(s, unicode):
            string = string.encode('utf-8')
        return [ord(c) + numres for c in string]

    r = [c + numres for c in string.encode('utf-8')]
    if add_eos:
        r = r + [1]
    return r


def decode(ids, strip_extraneous = False):
    """
    Decode integer representation to string based on ascii table.

    Parameters
    -----------
    ids: List[int]

    Returns
    --------
    result: str
    """
    if strip_extraneous:
        ids = strip_ids(ids, list(range(NUM_RESERVED_TOKENS or 0)))
    numres = NUM_RESERVED_TOKENS
    decoded_ids = []
    int2byte = six.int2byte
    for id_ in ids:
        if 0 <= id_ < numres:
            decoded_ids.append(RESERVED_TOKENS_BYTES[int(id_)])
        else:
            decoded_ids.append(int2byte(id_ - numres))
    if six.PY2:
        return ''.join(decoded_ids)
    return b''.join(decoded_ids).decode('utf-8', 'replace')
