from malaya_speech.supervised import stt
from herpetologist import check_type
from malaya_speech.utils import describe_availability
import json

_transducer_availability = {
    'tiny-conformer': {
        'Size (MB)': 24.4,
        'Quantized Size (MB)': 9.14,
        'WER': 0.2128108,
        'CER': 0.08136871,
        'WER-LM': 0.1996828,
        'CER-LM': 0.0770037,
        'Language': ['malay'],
    },
    'small-conformer': {
        'Size (MB)': 49.2,
        'Quantized Size (MB)': 18.1,
        'WER': 0.19853302,
        'CER': 0.07449528,
        'WER-LM': 0.18536073,
        'CER-LM': 0.07114307,
        'Language': ['malay'],
    },
    'conformer': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 37.1,
        'WER': 0.1636023,
        'CER': 0.0587443,
        'WER-LM': 0.1561821,
        'CER-LM': 0.0571897,
        'Language': ['malay'],
    },
    'large-conformer': {
        'Size (MB)': 404,
        'Quantized Size (MB)': 107,
        'WER': 0.1566839,
        'CER': 0.0619715,
        'WER-LM': 0.1486221,
        'CER-LM': 0.0590102,
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
        'WER': 0.19809173,
        'CER': 0.07903460,
        'WER-LM': 0.19884239,
        'CER-LM': 0.07812183,
        'Language': ['malay'],
    },
    'sm-squeezeformer': {
        'Size (MB)': 147,
        'Quantized Size (MB)': 47.4,
        'WER': 0.176126847,
        'CER': 0.0680792,
        'WER-LM': 0.1687297,
        'CER-LM': 0.061468,
        'Language': ['malay'],
    },
    'm-squeezeformer': {
        'Size (MB)': 261,
        'Quantized Size (MB)': 78.5,
        'WER': 0.16700751,
        'CER': 0.05972837,
        'WER-LM': 0.15618489,
        'CER-LM': 0.05363883,
        'Language': ['malay'],
    },
}

_ctc_availability = {
    'hubert-conformer-tiny': {
        'Size (MB)': 36.6,
        'Quantized Size (MB)': 10.3,
        'WER': 0.3359682,
        'CER': 0.0882573,
        'WER-LM': 0.1992265,
        'CER-LM': 0.0635223,
        'Language': ['malay'],
    },
    'hubert-conformer': {
        'Size (MB)': 115,
        'Quantized Size (MB)': 31.1,
        'WER': 0.238714,
        'CER': 0.0608998,
        'WER-LM': 0.1414791,
        'CER-LM': 0.0450751,
        'Language': ['malay'],
    },
    'hubert-conformer-large': {
        'Size (MB)': 392,
        'Quantized Size (MB)': 100,
        'WER': 0.2203140,
        'CER': 0.0549270,
        'WER-LM': 0.1280064,
        'CER-LM': 0.03853289,
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
        'WER': 0.3192907,
        'CER': 0.078988,
        'WER-LM': 0.179582,
        'CER-LM': 0.055521,
        'Language': ['malay'],
    },
    'best-rq-conformer': {
        'Size (MB)': 115,
        'Quantized Size (MB)': 31.1,
        'WER': 0.2536784,
        'CER': 0.0658045,
        'WER-LM': 0.1542058,
        'CER-LM': 0.0482278,
        'Language': ['malay'],
    },
    'best-rq-conformer-large': {
        'Size (MB)': 392,
        'Quantized Size (MB)': 100,
        'WER': 0.2346511,
        'CER': 0.0601605,
        'WER-LM': 0.1300819,
        'CER-LM': 0.044521,
        'Language': ['malay'],
    },
}

# https://huggingface.co/mesolitica/wav2vec2-xls-r-300m-mixed
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

google_accuracy = {
    'malay': {
        'WER': 0.164775,
        'CER': 0.0597320,
    },
    'singlish': {
        'WER': 0.4941349,
        'CER': 0.3026296,
    }
}


def available_ctc():
    """
    List available Encoder-CTC ASR models.
    """

    return describe_availability(_ctc_availability)


def available_transducer():
    """
    List available Encoder-Transducer ASR models.
    """

    return describe_availability(_transducer_availability)


def available_huggingface():
    """
    List available HuggingFace Malaya-Speech ASR models.
    """

    return describe_availability(_huggingface_availability)


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
def huggingface(model: str = 'mesolitica/wav2vec2-xls-r-300m-mixed', **kwargs):
    """
    Load Finetuned models from HuggingFace. Required Tensorflow >= 2.0.

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
