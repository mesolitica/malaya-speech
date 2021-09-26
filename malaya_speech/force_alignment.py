from malaya_speech.supervised import force_alignment
from herpetologist import check_type

_availability = {
    'fastspeech-decoder-aligner-tts': {
        'Size (MB)': 49.2,
        'Quantized Size (MB)': 18.1,
        'Language': ['malay'],
    },
    'fastspeech-decoder-aligner-stt': {
        'Size (MB)': 49.2,
        'Quantized Size (MB)': 18.1,
        'Language': ['malay'],
    },
    'fastspeech-decoder-aligner-mixed': {
        'Size (MB)': 49.2,
        'Quantized Size (MB)': 18.1,
        'Language': ['mixed'],
    },
    'fastspeech-decoder-aligner-singlish': {
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
    model: str = 'fastspeech-decoder-aligner-stt', quantized: bool = False, **kwargs
):
    """
    Load Deep Aligner model.

    Parameters
    ----------
    model : str, optional (default='fastspeech-decoder-aligner-stt')
        Model architecture supported. Allowed values:

        * ``'fastspeech-decoder-aligner-tts'`` - FastSpeech Decoder trained on Malay TTS dataset.
        * ``'fastspeech-decoder-aligner-stt'`` - FastSpeech Decoder trained on Malay STT dataset.
        * ``'fastspeech-decoder-aligner-mixed'`` - FastSpeech Decoder trained on Mixed STT dataset.
        * ``'fastspeech-decoder-aligner-singlish'`` - FastSpeech Decoder trained on Singlish STT dataset.

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
