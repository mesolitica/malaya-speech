from malaya_speech.supervised import stt
from herpetologist import check_type
from malaya_speech.utils import describe_availability
from malaya_speech.stt import _describe
import warnings

_transformer_availability = {
    'hubert-conformer-tiny': {
        'Size (MB)': 36.6,
        'Quantized Size (MB)': 10.3,
        'malay-malaya': {
            'WER': 0.238714008166,
            'CER': 0.060899814,
            'WER-LM': 0.14147911604,
            'CER-LM': 0.04507517237,
        },
        'Language': ['malay'],
    },
    'hubert-conformer': {
        'Size (MB)': 115,
        'Quantized Size (MB)': 31.1,
        'malay-malaya': {
            'WER': 0.2387140081,
            'CER': 0.06089981404,
            'WER-LM': 0.141479116045,
            'CER-LM': 0.0450751723784,
        },
        'Language': ['malay'],
    },
    'hubert-conformer-large': {
        'Size (MB)': 392,
        'Quantized Size (MB)': 100,
        'malay-malaya': {
            'WER': 0.2203140421,
            'CER': 0.0549270416,
            'WER-LM': 0.1280064575,
            'CER-LM': 0.03853289571,
        },
        'Language': ['malay'],
    },
}

_huggingface_availability = {
    'mesolitica/wav2vec2-xls-r-300m-mixed': {
        'Size (MB)': 1180,
        'malay-malaya': {
            'WER': 0.194655128,
            'CER': 0.04775798,
            'WER-LM': 0.12849904267,
            'CER-LM': 0.0357602212,
        },
        'malay-fleur102': {
            'WER': 0.2373861259,
            'CER': 0.0705547800,
            'WER-LM': 0.17169389543,
            'CER-LM': 0.0591631316,
        },
        'singlish': {
            'WER': 0.127588595,
            'CER': 0.0494924979,
            'WER-LM': 0.096820291,
            'CER-LM': 0.042727603,
        },
        'Language': ['malay', 'singlish'],
    },
    'mesolitica/wav2vec2-xls-r-300m-mixed-v2': {
        'Size (MB)': 1180,
        'malay-malaya': {
            'WER': 0.154782923,
            'CER': 0.035164031,
            'WER-LM': 0.09177104918,
            'CER-LM': 0.0253532288,
        },
        'malay-fleur102': {
            'WER': 0.2013994374,
            'CER': 0.0518170369,
            'WER-LM': 0.14364611216,
            'CER-LM': 0.0416905231,
        },
        'singlish': {
            'WER': 0.2258822139,
            'CER': 0.082982312,
            'WER-LM': 0.17862153528,
            'CER-LM': 0.0718263800,
        },
        'Language': ['malay', 'singlish'],
    },
    'mesolitica/wav2vec2-xls-r-300m-12layers-ms': {
        'Size (MB)': 657,
        'malay-malaya': {
            'WER': 0.1494983789,
            'CER': 0.0342059992,
            'WER-LM': 0.08810564526,
            'CER-LM': 0.02455549021,
        },
        'malay-fleur102': {
            'WER': 0.217107489,
            'CER': 0.0546614199,
            'WER-LM': 0.1481601917,
            'CER-LM': 0.04311461708,
        },
        'Language': ['malay'],
    },
    'mesolitica/wav2vec2-xls-r-300m-6layers-ms': {
        'Size (MB)': 339,
        'malay-malaya': {
            'WER': 0.22481538553,
            'CER': 0.0484392694,
            'WER-LM': 0.12168497766,
            'CER-LM': 0.03329379029,
        },
        'malay-fleur102': {
            'WER': 0.38642364985,
            'CER': 0.0928960677,
            'WER-LM': 0.2896173391,
            'CER-LM': 0.072466167,
        },
        'Language': ['malay'],
    },
}


def available_transformer():
    """
    List available Encoder-CTC ASR models.
    """

    warnings.warn(
        '`malaya.stt.ctc.available_transformer` is deprecated, use `malaya.stt.ctc.available_huggingface` instead',
        DeprecationWarning)
    _describe()
    return describe_availability(_transformer_availability)


def available_huggingface():
    """
    List available HuggingFace CTC ASR models.
    """

    _describe()
    return describe_availability(_huggingface_availability)


@check_type
def transformer(
    model: str = 'hubert-conformer',
    quantized: bool = False,
    **kwargs,
):
    """
    Load Encoder-CTC ASR model.

    Parameters
    ----------
    model : str, optional (default='hubert-conformer')
        Check available models at `malaya_speech.stt.ctc.available_transformer()`.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.model.wav2vec.Wav2Vec2_CTC class
    """
    warnings.warn(
        '`malaya.stt.ctc.transformer` is deprecated, use `malaya.stt.ctc.huggingface` instead', DeprecationWarning)

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.stt.ctc.available_transformer()`.'
        )

    return stt.wav2vec2_ctc_load(
        model=model,
        module='speech-to-text-ctc-v2',
        quantized=quantized,
        mode=_transformer_availability[model],
        **kwargs
    )


@check_type
def huggingface(
    model: str = 'mesolitica/wav2vec2-xls-r-300m-mixed',
    force_check: bool = True,
    **kwargs,
):
    """
    Load Finetuned models from HuggingFace.

    Parameters
    ----------
    model : str, optional (default='mesolitica/wav2vec2-xls-r-300m-mixed')
        Check available models at `malaya_speech.stt.ctc.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result : malaya_speech.torch_model.huggingface.CTC class
    """
    model = model.lower()
    if model not in _huggingface_availability and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.stt.ctc.available_huggingface()`.'
        )

    return stt.huggingface_load(model=model, **kwargs)
