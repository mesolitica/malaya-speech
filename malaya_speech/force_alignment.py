from malaya_speech.supervised import stt
from malaya_speech.stt import _ctc_availability, _huggingface_availability
from malaya_speech.utils import describe_availability
from herpetologist import check_type

_availability = {
    'conformer-transducer': {
        'Size (MB)': 120,
        'Quantized Size (MB)': 32.3,
        'Language': ['malay'],
    },
    'conformer-transducer-mixed': {
        'Size (MB)': 120,
        'Quantized Size (MB)': 32.3,
        'Language': ['malay', 'singlish'],
    },
    'conformer-transducer-singlish': {
        'Size (MB)': 120,
        'Quantized Size (MB)': 32.3,
        'Language': ['singlish'],
    },
}


def available_transducer():
    """
    List available Encoder-Transducer Aligner models.
    """

    return describe_availability(_availability)


def available_ctc():
    """
    List available Encoder-CTC Aligner models.
    """

    return describe_availability(_ctc_availability)


def available_huggingface():
    """
    List available HuggingFace Malaya-Speech Aligner models.
    """

    return describe_availability(_huggingface_availability)


@check_type
def deep_transducer(
    model: str = 'conformer-transducer', quantized: bool = False, **kwargs
):
    """
    Load Encoder-Transducer Aligner model.

    Parameters
    ----------
    model : str, optional (default='conformer-transducer')
        Check available models at `malaya_speech.force_alignment.available_aligner()`.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.model.transducer.TransducerAligner class
    """
    model = model.lower()
    if model not in _availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.force_alignment.available_aligner()`.'
        )

    return stt.transducer_load(
        model=model,
        module='force-alignment',
        languages=_availability[model]['Language'],
        quantized=quantized,
        stt=False,
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
    result : malaya_speech.model.wav2vec.Wav2Vec2_Aligner class
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
        stt=False,
        **kwargs
    )


@check_type
def huggingface(model: str = 'mesolitica/wav2vec2-xls-r-300m-mixed'):
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

    return stt.huggingface_load(model=model, stt=False)
