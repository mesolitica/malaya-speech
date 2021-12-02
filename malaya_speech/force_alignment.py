from malaya_speech.supervised import stt
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


def available_aligner():
    """
    List available Deep Aligner models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(_availability)


@check_type
def deep_aligner(
    model: str = 'conformer-transducer', quantized: bool = False, **kwargs
):
    """
    Load Deep Aligner model.

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
    result : malaya_speech.model.tf.TransducerAligner class
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
