from malaya_speech.path import (
    PATH_STT_CTC,
    S3_PATH_STT_CTC,
    PATH_LM,
    S3_PATH_LM,
)
from malaya_speech.supervised import stt
from herpetologist import check_type
import json

_transducer_availability = {
    'small-conformer': {'Size (MB)': 97.8, 'WER': 0, 'CER': 0},
    'base-conformer': {'Size (MB)': 97.8, 'WER': 0, 'CER': 0},
}

_ctc_availability = {
    'quartznet': {
        'Size (MB)': 77.2,
        'Quantized Size (MB)': 20.2,
        'WER': 0,
        'CER': 0,
        'WER-LM': 0,
        'CER-LM': 0,
    },
    'mini-jasper': {
        'Size (MB)': 97.8,
        'Quantized Size (MB)': 20.2,
        'WER': 0,
        'CER': 0,
        'WER-LM': 0,
        'CER-LM': 0,
    },
    'jasper': {
        'Size (MB)': 97.8,
        'Quantized Size (MB)': 20.2,
        'WER': 0,
        'CER': 0,
        'WER-LM': 0,
        'CER-LM': 0,
    },
}

_language_model_availability = {
    'malaya-speech': {
        'Size (MB)': 2.8,
        'Description': 'Gathered from malaya-speech ASR transcript',
        'Command': [
            'lmplz --text text.txt --arpa out.arpa -o 3 --prune 0 1 1',
            'build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
        ],
    },
    'malaya-speech-wikipedia': {
        'Size (MB)': 23.3,
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
        * ``'local'`` - Gathered from IIUM Confession.
        
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
        vocab_list = json.load(fopen)

    scorer = Scorer(alpha, beta, PATH_LM[model]['model'], vocab_list)
    return scorer


def deep_transducer(
    model: str = 'base-conformer', quantized: bool = False, **kwargs
):
    """
    Load Encoder-Transducer ASR model.

    Parameters
    ----------
    model : str, optional (default='jasper')
        Model architecture supported. Allowed values:

        * ``'small-conformer'`` - SMALL size Google Conformer, https://arxiv.org/pdf/2005.08100.pdf
        * ``'base-conformer'`` - BASE size Google Conformer, https://arxiv.org/pdf/2005.08100.pdf
        
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model. 
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.supervised.classification.load function
    """
    model = model.lower()
    if model not in _transducer_availability:
        raise Exception(
            'model not supported, please check supported models from `malaya_speech.stt.available_transducer()`.'
        )


def deep_ctc(model: str = 'jasper', quantized: bool = False, **kwargs):
    """
    Load Encoder-CTC ASR model.

    Parameters
    ----------
    model : str, optional (default='jasper')
        Model architecture supported. Allowed values:

        * ``'quartznet'`` - NVIDIA QuartzNet, https://arxiv.org/abs/1910.10261
        * ``'mini-jasper'`` - Small-factor NVIDIA Jasper, https://arxiv.org/pdf/1904.03288.pdf
        * ``'jasper'`` - NVIDIA Jasper, https://arxiv.org/pdf/1904.03288.pdf
        
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model. 
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.supervised.stt.load function
    """

    model = model.lower()
    if model not in _ctc_availability:
        raise Exception(
            'model not supported, please check supported models from `malaya_speech.stt.available_ctc()`.'
        )

    return stt.load(
        path = PATH_STT_CTC,
        s3_path = S3_PATH_STT_CTC,
        model = model,
        name = 'speech-to-text',
        quantized = quantized,
        **kwargs
    )
