from malaya_speech.supervised import stt
from malaya_speech.stt import _ctc_availability
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
        'Language': ['mixed'],
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
    from malaya_speech.utils import describe_availability

    return describe_availability(_availability)


def available_ctc():
    """
    List available Encoder-CTC Aligner models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(_ctc_availability)


@check_type
def deep_transducer(
    model: str = 'conformer-transducer', quantized: bool = False, **kwargs
):
    """
    Load Encoder-Transducer Aligner model.

    Parameters
    ----------
    model : str, optional (default='conformer-transducer')
        Model architecture supported. Allowed values:

        * ``'conformer-transducer'`` - Conformer + RNNT trained on Malay STT dataset.
        * ``'conformer-transducer-mixed'`` - Conformer + RNNT trained on Mixed STT dataset.
        * ``'conformer-transducer-singlish'`` - Conformer + RNNT trained on Singlish STT dataset.

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
        Model architecture supported. Allowed values:

        * ``'hubert-conformer-tiny'`` - Finetuned HuBERT Conformer TINY.
        * ``'hubert-conformer'`` - Finetuned HuBERT Conformer.
        * ``'hubert-conformer-large'`` - Finetuned HuBERT Conformer LARGE.
        * ``'hubert-conformer-large-3mixed'`` - Finetuned HuBERT Conformer LARGE for (Malay + Singlish + Mandarin) languages.
        * ``'best-rq-conformer-tiny'`` - Finetuned BEST-RQ Conformer TINY.
        * ``'best-rq-conformer'`` - Finetuned BEST-RQ Conformer.
        * ``'best-rq-conformer-large'`` - Finetuned BEST-RQ Conformer LARGE.


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
