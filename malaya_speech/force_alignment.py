from malaya_speech.supervised import force_alignment
from herpetologist import check_type

_availability = {
    'small-conformer-transducer-tts': {
        'Size (MB)': 49.2,
        'Quantized Size (MB)': 18.1,
        'Language': ['malay'],
    },
    'conformer-transducer-stt': {
        'Size (MB)': 49.2,
        'Quantized Size (MB)': 18.1,
        'Language': ['malay'],
    },
    'conformer-transducer-mixed': {
        'Size (MB)': 49.2,
        'Quantized Size (MB)': 18.1,
        'Language': ['mixed'],
    },
    'conformer-transducer-singlish': {
        'Size (MB)': 49.2,
        'Quantized Size (MB)': 18.1,
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
    model: str = 'small-conformer-transducer-tts', quantized: bool = False, **kwargs
):
    """
    Load Deep Aligner model.

    Parameters
    ----------
    model : str, optional (default='small-conformer-transducer-tts')
        Model architecture supported. Allowed values:

        * ``'small-conformer-transducer-tts'`` - Small Conformer + RNNT trained on Malay TTS dataset.
        * ``'conformer-transducer-stt'`` - Conformer + RNNT trained on Malay STT dataset.
        * ``'conformer-transducer-mixed'`` - Conformer + RNNT trained on Mixed STT dataset.
        * ``'conformer-transducer-singlish'`` - Conformer + RNNT trained on Singlish STT dataset.

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.model.tf.DeepAligner class
    """
    model = model.lower()
    if model not in _availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.force_alignment.available_aligner()`.'
        )
