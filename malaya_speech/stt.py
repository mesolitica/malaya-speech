from malaya_speech.path import (
    PATH_STT_CTC,
    S3_PATH_STT_CTC,
    PATH_STT_TRANSDUCER,
    S3_PATH_STT_TRANSDUCER,
    PATH_LM,
    S3_PATH_LM,
)
from malaya_speech.supervised import stt
from herpetologist import check_type
import json

_transducer_availability = {
    'small-conformer': {
        'Size (MB)': 44.1,
        'Quantized Size (MB)': 13.3,
        'WER': 0.2692,
        'CER': 0.1058,
    },
    'conformer': {
        'Size (MB)': 120,
        'Quantized Size (MB)': 32.7,
        'WER': 0.2442,
        'CER': 0.0910,
    },
    'large-conformer': {
        'Size (MB)': 399,
        'Quantized Size (MB)': 103,
        'WER': 0.2390,
        'CER': 0.0812,
    },
    'alconformer': {
        'Size (MB)': 33.2,
        'Quantized Size (MB)': 10.5,
        'WER': 0.30567,
        'CER': 0.12267,
    },
}

_ctc_availability = {
    'mini-jasper': {
        'Size (MB)': 33.3,
        'Quantized Size (MB)': 8.71,
        'WER': 0.3353,
        'CER': 0.0870,
    },
    'medium-jasper': {
        'Size (MB)': 336,
        'Quantized Size (MB)': 84.9,
        'WER': 0.3383,
        'CER': 0.0922,
    },
    'jasper': {
        'Size (MB)': 1290,
        'Quantized Size (MB)': 323,
        'WER': 0.3215,
        'CER': 0.0882,
    },
}

_language_model_availability = {
    'malaya-speech': {
        'Size (MB)': 4.5,
        'Description': 'Gathered from malaya-speech ASR transcript',
        'Command': [
            'lmplz --text text.txt --arpa out.arpa -o 3 --prune 0 1 1',
            'build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
        ],
    },
    'malaya-speech-wikipedia': {
        'Size (MB)': 27.5,
        'Description': 'Gathered from malaya-speech ASR transcript + Wikipedia (Random sample 300k sentences)',
        'Command': [
            'lmplz --text text.txt --arpa out.arpa -o 5 --prune 0 1 1 1 1',
            'build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
        ],
    },
    'local': {
        'Size (MB)': 22,
        'Description': 'Gathered from IIUM Confession',
        'Command': [
            'lmplz --text text.txt --arpa out.arpa -o 5 --prune 0 1 1 1 1',
            'build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
        ],
    },
    'wikipedia': {
        'Size (MB)': 95.6,
        'Description': 'Gathered from malay Wikipedia',
        'Command': [
            'lmplz --text text.txt --arpa out.arpa -o 5 --prune 0 1 1 1 1',
            'build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
        ],
    },
}


def available_transducer():
    """
    List available Encoder-Transducer ASR models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(_transducer_availability)


def available_ctc():
    """
    List available Encoder-CTC ASR models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(_ctc_availability)


def available_language_model():
    """
    List available Language Model for CTC.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(_language_model_availability)


@check_type
def language_model(
    model: str = 'malaya-speech',
    alpha: float = 2.5,
    beta: float = 0.3,
    **kwargs
):
    """
    Load KenLM language model.

    Parameters
    ----------
    model : str, optional (default='malaya-speech')
        Model architecture supported. Allowed values:

        * ``'malaya-speech'`` - Gathered from malaya-speech ASR transcript.
        * ``'malaya-speech-wikipedia'`` - Gathered from malaya-speech ASR transcript + Wikipedia (Random sample 300k sentences).
        * ``'local'`` - Gathered from IIUM Confession.
        * ``'wikipedia'`` - Gathered from malay Wikipedia.
        
    alpha: float, optional (default=2.5)
        score = alpha * np.log(lm) + beta * np.log(word_cnt), 
        increase will put more bias on lm score computed by kenlm.
    beta: float, optional (beta=0.3)
        score = alpha * np.log(lm) + beta * np.log(word_cnt), 
        increase will put more bias on word count.

    Returns
    -------
    result : Tuple[ctc_decoders.Scorer, List[str]]
        Tuple of ctc_decoders.Scorer and vocab.
    """
    try:
        from ctc_decoders import Scorer
    except:
        raise ModuleNotFoundError(
            'ctc_decoders not installed. Please install it by `pip install ctc-decoders` and try again.'
        )
    from malaya_speech.utils import check_file

    check_file(PATH_LM[model], S3_PATH_LM[model], **kwargs)

    with open(PATH_LM[model]['vocab']) as fopen:
        vocab_list = json.load(fopen) + ['{', '}', '[']

    scorer = Scorer(alpha, beta, PATH_LM[model]['model'], vocab_list)
    return scorer


@check_type
def deep_transducer(
    model: str = 'conformer', quantized: bool = False, **kwargs
):
    """
    Load Encoder-Transducer ASR model.

    Parameters
    ----------
    model : str, optional (default='jasper')
        Model architecture supported. Allowed values:

        * ``'small-conformer'`` - SMALL size Google Conformer, https://arxiv.org/pdf/2005.08100.pdf
        * ``'conformer'`` - BASE size Google Conformer, https://arxiv.org/pdf/2005.08100.pdf
        * ``'alconformer'`` - BASE size A-Lite Google Conformer.
        
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model. 
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.supervised.stt.transducer_load function
    """
    model = model.lower()
    if model not in _transducer_availability:
        raise Exception(
            'model not supported, please check supported models from `malaya_speech.stt.available_transducer()`.'
        )

    return stt.transducer_load(
        path = PATH_STT_TRANSDUCER,
        s3_path = S3_PATH_STT_TRANSDUCER,
        model = model,
        name = 'speech-to-text',
        quantized = quantized,
        **kwargs
    )


@check_type
def deep_ctc(model: str = 'jasper', quantized: bool = False, **kwargs):
    """
    Load Encoder-CTC ASR model.

    Parameters
    ----------
    model : str, optional (default='jasper')
        Model architecture supported. Allowed values:

        * ``'mini-jasper'`` - Small-factor NVIDIA Jasper, https://arxiv.org/pdf/1904.03288.pdf
        * ``'medium-jasper'`` - Medium-factor NVIDIA Jasper, https://arxiv.org/pdf/1904.03288.pdf
        * ``'jasper'`` - NVIDIA Jasper, https://arxiv.org/pdf/1904.03288.pdf
        
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model. 
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.supervised.stt.ctc_load function
    """

    model = model.lower()
    if model not in _ctc_availability:
        raise Exception(
            'model not supported, please check supported models from `malaya_speech.stt.available_ctc()`.'
        )

    return stt.ctc_load(
        path = PATH_STT_CTC,
        s3_path = S3_PATH_STT_CTC,
        model = model,
        name = 'speech-to-text',
        quantized = quantized,
        **kwargs
    )
