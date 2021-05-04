from malaya_speech.utils import check_file
from malaya_speech.path import TRANSDUCER_VOCABS, CTC_VOCABS
import json


def get_vocab_ctc(language):
    return CTC_VOCABS.get(language, CTC_VOCABS['malay'])


def load(model, module, alpha, beta, **kwargs):
    path = check_file(
        file = model,
        module = module,
        keys = {
            'model': 'model.trie.klm',
            'vocab': get_vocab_ctc(model.split('-')[-1]),
        },
        quantized = False,
        **kwargs,
    )

    try:
        from ctc_decoders import Scorer
    except:
        raise ModuleNotFoundError(
            'ctc_decoders not installed. Please install it by `pip3 install ctc-decoders` and try again.'
        )

    with open(path['vocab']) as fopen:
        vocab_list = json.load(fopen) + ['{', '}', '[']

    scorer = Scorer(alpha, beta, path['model'], vocab_list)
    return scorer
