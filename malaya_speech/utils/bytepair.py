import os


class BYTEPAIR:
    def __init__(self, model, encode_type):
        self.model = model
        self._encode_type = encode_type

    def encode(self, string):
        """
        Encode string to integer representation.

        Parameters
        -----------
        string: str

        Returns
        --------
        result: List[int]
        """
        return self.model.encode(string, output_type = self._encode_type) + [1]

    def decode(self, ids):
        """
        Decode integer representation to string.

        Parameters
        -----------
        ids: List[int]

        Returns
        --------
        result: str
        """
        ids = [i for i in ids if i > 1]
        return self.model.decode(ids)[0]


def train(text, model_path: str = 'bpe.model', vocab_size: int = 200):
    """
    Train YouTokenToMe bytepair encoding.

    Parameters
    ----------
    text: List[str] / Tuple[str] / str
        Can be List[str] or Tuple[str] or file name.
    model_path: str, optional (default='bpe.model')
        model name.
    vocab_size: int, optional (default=200)

    Returns
    -------
    result : yttm.BPE
    """

    try:
        import youtokentome as yttm
    except:
        raise ValueError(
            'youtokentome not installed. Please install it by `pip install youtokentome` and try again.'
        )

    delete = False
    if isinstance(text, list) or isinstance(tuple):
        delete = True

        with open('temp.txt', 'w') as fopen:
            fopen.write('\n'.join(text))

        text = 'temp.txt'

    bpe = yttm.BPE.train(
        data = text,
        vocab_size = vocab_size,
        model = model_path,
        pad_id = 0,
        unk_id = 3,
        bos_id = 2,
        eos_id = 1,
    )

    if delete:
        try:
            os.remove(text)
        except:
            pass

    return bpe


def load(model_path: str):
    """
    Load YouTokenToMe bytepair encoding.

    Parameters
    ----------
    model_path: str
        model name.

    Returns
    -------
    result : yttm.BPE
    """
    try:
        import youtokentome as yttm
    except:
        raise ValueError(
            'youtokentome not installed. Please install it by `pip install youtokentome` and try again.'
        )

    return BYTEPAIR(yttm.BPE(model = model_path), yttm.OutputType.ID)
