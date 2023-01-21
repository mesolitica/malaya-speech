from malaya_speech.supervised import stt
from herpetologist import check_type
from malaya_speech.utils import describe_availability
from malaya_speech.stt import _describe
import warnings

_transformer_availability = {
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
    'mesolitica/wav2vec2-xls-r-1b-01-ms': {
        'Size (MB)': 230,
        'WER': 0.1322198,
        'CER': 0.0481054,
        'WER-LM': 0.0988016,
        'CER-LM': 0.0411965,
        'Language': ['malay'],
    }
    'mesolitica/wav2vec2-xls-r-1b-0123-ms': {
        'Size (MB)': 387,
        'WER': 0.1322198,
        'CER': 0.0481054,
        'WER-LM': 0.0988016,
        'CER-LM': 0.0411965,
        'Language': ['malay'],
    },
    'mesolitica/wav2vec2-xls-r-1b-012345-ms': {
        'Size (MB)': 544,
        'WER': 0.1322198,
        'CER': 0.0481054,
        'WER-LM': 0.0988016,
        'CER-LM': 0.0411965,
        'Language': ['malay'],
    }
}


def available_transformer():
    """
    List available Encoder-CTC ASR models.
    """

    warnings.warn(
        '`malaya.stt.ctc.available_transformer` is deprecated, use `malaya.stt.ctc.available_huggingface` instead', DeprecationWarning)
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

    Returns
    -------
    result : malaya_speech.model.huggingface.CTC class
    """
    model = model.lower()
    if model not in _huggingface_availability and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.stt.ctc.available_huggingface()`.'
        )

    return stt.huggingface_load(model=model, **kwargs)
