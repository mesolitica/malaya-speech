import os


def train(text, model_path: str, vocab_size: int = 500):
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

    return yttm.BPE.train(
        data = text, vocab_size = vocab_size, model = model_path
    )


class BYTEPAIR:
    def __init__(self, model, encode_type):
        self.model = model
        self._encode_type = encode_type

    def encode(self, string):
        return self.model.encode(string, output_type = self._encode_type)

    def decode(self, ids):
        return self.model.decode(ids)


def load(model_path):
    try:
        import youtokentome as yttm
    except:
        raise ValueError(
            'youtokentome not installed. Please install it by `pip install youtokentome` and try again.'
        )

    return BYTEPAIR(yttm.BPE(model = model_path), yttm.OutputType.ID)
