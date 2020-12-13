_availability = {
    'melgan-male': {
        'Size (MB)': 20,
        'Quantized Size (MB)': 13.3,
        'STFT loss': 0,
        'Mel loss': 0,
    },
    'melgan-female': {
        'Size (MB)': 20,
        'Quantized Size (MB)': 13.3,
        'STFT loss': 0,
        'Mel loss': 0,
    },
    'mbmelgan-male': {
        'Size (MB)': 20,
        'Quantized Size (MB)': 13.3,
        'STFT loss': 0,
        'Mel loss': 0,
    },
    'mbmelgan-female': {
        'Size (MB)': 20,
        'Quantized Size (MB)': 13.3,
        'STFT loss': 0,
        'Mel loss': 0,
    },
}


def available_model():
    """
    List available Vocoder deep models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(_availability)


def deep_model(model: str = 'melgan-male', quantized = True, **kwargs):
    if model not in _availability:
        raise Exception(
            'model not supported, please check supported models from `malaya_speech.vocoder.available_model()`.'
        )
