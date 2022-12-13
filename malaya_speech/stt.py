from malaya_speech.supervised import stt
from herpetologist import check_type
from malaya_speech.utils import describe_availability
import json
import logging

logger = logging.getLogger(__name__)

_transducer_availability = {
    'tiny-conformer': {
        'Size (MB)': 24.4,
        'Quantized Size (MB)': 9.14,
        'WER': 0.25205879818082805,
        'CER': 0.11939432658062442,
        'WER-LM': 0.24466923675904093,
        'CER-LM': 0.12099917108722165,
        'Language': ['malay'],
    },
    'small-conformer': {
        'Size (MB)': 49.2,
        'Quantized Size (MB)': 18.1,
        'WER': 0.2216678092407059,
        'CER': 0.10721028207884116,
        'WER-LM': 0.215915398935267,
        'CER-LM': 0.10903240184809346,
        'Language': ['malay'],
    },
    'conformer': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 37.1,
        'WER': 0.1854134610718644,
        'CER': 0.08612027714394176,
        'WER-LM': 0.18542071537532542,
        'CER-LM': 0.09424689146824226,
        'Language': ['malay'],
    },
    'large-conformer': {
        'Size (MB)': 404,
        'Quantized Size (MB)': 107,
        'WER': 0.16068731055857535,
        'CER': 0.07225488182147317,
        'WER-LM': 0.15852097352196282,
        'CER-LM': 0.07445114519781025,
        'Language': ['malay'],
    },
    'conformer-stack-2mixed': {
        'Size (MB)': 130,
        'Quantized Size (MB)': 38.5,
        'WER': 0.1036084,
        'CER': 0.050069,
        'WER-LM': 0.1029111,
        'CER-LM': 0.0502013,
        'Language': ['malay', 'singlish'],
    },
    'conformer-stack-3mixed': {
        'Size (MB)': 130,
        'Quantized Size (MB)': 38.5,
        'WER': 0.2347684,
        'CER': 0.133944,
        'WER-LM': 0.229241,
        'CER-LM': 0.1307018,
        'Language': ['malay', 'singlish', 'mandarin'],
    },
    'small-conformer-singlish': {
        'Size (MB)': 49.2,
        'Quantized Size (MB)': 18.1,
        'WER': 0.0878310,
        'CER': 0.0456859,
        'WER-LM': 0.08733263,
        'CER-LM': 0.04531657,
        'Language': ['singlish'],
    },
    'conformer-singlish': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 37.1,
        'WER': 0.07779246,
        'CER': 0.0403616,
        'WER-LM': 0.07718602,
        'CER-LM': 0.03986952,
        'Language': ['singlish'],
    },
    'large-conformer-singlish': {
        'Size (MB)': 404,
        'Quantized Size (MB)': 107,
        'WER': 0.07014733,
        'CER': 0.03587201,
        'WER-LM': 0.06981206,
        'CER-LM': 0.03572307,
        'Language': ['singlish'],
    },
    'xs-squeezeformer': {
        'Size (MB)': 51.9,
        'Quantized Size (MB)': 23.4,
        'WER': 0.24361950288615478,
        'CER': 0.1101217363425894,
        'WER-LM': 0.23732601782189705,
        'CER-LM': 0.11326849769270286,
        'Language': ['malay'],
    },
    'sm-squeezeformer': {
        'Size (MB)': 147,
        'Quantized Size (MB)': 47.4,
        'WER': 0.1995528388961415,
        'CER': 0.09630517271889948,
        'WER-LM': 0.1942312047785515,
        'CER-LM': 0.09945312677604806,
        'Language': ['malay'],
    },
    'm-squeezeformer': {
        'Size (MB)': 261,
        'Quantized Size (MB)': 78.5,
        'WER': 0.19985013387051873,
        'CER': 0.09779469719228841,
        'WER-LM': 0.19800923596045886,
        'CER-LM': 0.10461433715575515,
        'Language': ['malay'],
    },
}

_ctc_availability = {
    'hubert-conformer-tiny': {
        'Size (MB)': 36.6,
        'Quantized Size (MB)': 10.3,
        'WER': 0.49570058359782315,
        'CER': 0.1601760228347541,
        'WER-LM': 0.4545267649665263,
        'CER-LM': 0.13999296317727367,
        'Language': ['malay'],
    },
    'hubert-conformer': {
        'Size (MB)': 115,
        'Quantized Size (MB)': 31.1,
        'WER': 0.33424915929720345,
        'CER': 0.10940574508444094,
        'WER-LM': 0.2710959758526313,
        'CER-LM': 0.09629870129384599,
        'Language': ['malay'],
    },
    'hubert-conformer-large': {
        'Size (MB)': 392,
        'Quantized Size (MB)': 100,
        'WER': 0.280827378386855,
        'CER': 0.09381837812108383,
        'WER-LM': 0.21454115630164386,
        'CER-LM': 0.08175489454799346,
        'Language': ['malay'],
    },
    'hubert-conformer-large-3mixed': {
        'Size (MB)': 392,
        'Quantized Size (MB)': 100,
        'WER': 0.2411256,
        'CER': 0.0787939,
        'WER-LM': 0.13276059,
        'CER-LM': 0.05748197,
        'Language': ['malay', 'singlish', 'mandarin'],
    },
    'best-rq-conformer-tiny': {
        'Size (MB)': 36.6,
        'Quantized Size (MB)': 10.3,
        'WER': 0.45708914980273974,
        'CER': 0.14645204643073154,
        'WER-LM': 0.421318523759614,
        'CER-LM': 0.1348827661179319,
        'Language': ['malay'],
    },
    'best-rq-conformer': {
        'Size (MB)': 115,
        'Quantized Size (MB)': 31.1,
        'WER': 0.3426014218184749,
        'CER': 0.11466136321470137,
        'WER-LM': 0.3286451554446768,
        'CER-LM': 0.11916163197851183,
        'Language': ['malay'],
    },
    'best-rq-conformer-large': {
        'Size (MB)': 392,
        'Quantized Size (MB)': 100,
        'WER': 0.2959243343164291,
        'CER': 0.1014638043480158,
        'WER-LM': 0.35880928946887464,
        'CER-LM': 0.12895209151263204,
        'Language': ['malay'],
    },
}

_huggingface_availability = {
    'mesolitica/wav2vec2-xls-r-300m-mixed': {
        'Size (MB)': 1180,
        'WER': 0.1322198,
        'CER': 0.0481054,
        'WER-LM': 0.0988016,
        'CER-LM': 0.0411965,
        'Language': ['malay', 'singlish', 'mandarin'],
    },
}

# https://github.com/huseinzol05/malaya-speech/blob/master/pretrained-model/prepare-stt/benchmark-google-speech-malay-dataset.ipynb
# https://github.com/huseinzol05/malaya-speech/blob/master/pretrained-model/prepare-stt/benchmark-google-speech-singlish-dataset.ipynb
google_accuracy = {
    'malay': {
        'WER': 0.109588779,
        'CER': 0.047891527,
    },
    'singlish': {
        'WER': 0.4941349,
        'CER': 0.3026296,
    }
}


def _describe():
    logger.info('for `malay` language, tested on FLEURS102 `ms_my` test set, https://github.com/huseinzol05/malaya-speech/tree/master/pretrained-model/prepare-stt')
    logger.info('for `singlish` language, tested on malaya-speech test set, https://github.com/huseinzol05/malaya-speech/tree/master/pretrained-model/prepare-stt')
    logger.info('for `mandarin` language, tested on malaya-speech test set, https://github.com/huseinzol05/malaya-speech/tree/master/pretrained-model/prepare-stt')


def available_transducer():
    """
    List available Encoder-Transducer ASR models.
    """

    _describe()
    return describe_availability(_transducer_availability)


def available_ctc():
    """
    List available Encoder-CTC ASR models.
    """

    _describe()
    return describe_availability(_ctc_availability)


def available_huggingface():
    """
    List available HuggingFace ASR models.
    """

    _describe()
    return describe_availability(_huggingface_availability)


@check_type
def deep_transducer(
    model: str = 'conformer', quantized: bool = False, **kwargs
):
    """
    Load Encoder-Transducer ASR model.

    Parameters
    ----------
    model : str, optional (default='conformer')
        Check available models at `malaya_speech.stt.available_transducer()`.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.model.transducer.Transducer class
    """
    model = model.lower()
    if model not in _transducer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.stt.available_transducer()`.'
        )

    return stt.transducer_load(
        model=model,
        module='speech-to-text-transducer',
        languages=_transducer_availability[model]['Language'],
        quantized=quantized,
        **kwargs
    )


@check_type
def deep_ctc(
    model: str = 'hubert-conformer', quantized: bool = False, **kwargs
):
    """
    Load Encoder-CTC ASR model.

    Parameters
    ----------
    model : str, optional (default='hubert-conformer')
        Check available models at `malaya_speech.stt.available_ctc()`.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.model.wav2vec.Wav2Vec2_CTC class
    """
    model = model.lower()
    if model not in _ctc_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.stt.available_ctc()`.'
        )

    return stt.wav2vec2_ctc_load(
        model=model,
        module='speech-to-text-ctc-v2',
        quantized=quantized,
        mode=_ctc_availability[model],
        **kwargs
    )


@check_type
def huggingface(model: str = 'mesolitica/wav2vec2-xls-r-300m-mixed', **kwargs):
    """
    Load Finetuned models from HuggingFace.

    Parameters
    ----------
    model : str, optional (default='mesolitica/wav2vec2-xls-r-300m-mixed')
        Check available models at `malaya_speech.stt.available_huggingface()`.
    Returns
    -------
    result : malaya_speech.model.huggingface.CTC class
    """
    model = model.lower()
    if model not in _huggingface_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.stt.available_huggingface()`.'
        )

    return stt.huggingface_load(model=model, **kwargs)
