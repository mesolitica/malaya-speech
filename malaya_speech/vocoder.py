_melgan_availability = {
    'male': {
        'Size (MB)': 20,
        'Quantized Size (MB)': 13.3,
        'STFT loss': 0,
        'Mel loss': 0,
    },
    'female': {
        'Size (MB)': 20,
        'Quantized Size (MB)': 13.3,
        'STFT loss': 0,
        'Mel loss': 0,
    },
    'husein': {
        'Size (MB)': 20,
        'Quantized Size (MB)': 13.3,
        'STFT loss': 0,
        'Mel loss': 0,
    },
}
_mbmelgan_availability = {
    'male': {
        'Size (MB)': 20,
        'Quantized Size (MB)': 13.3,
        'STFT loss': 0,
        'Mel loss': 0,
    },
    'female': {
        'Size (MB)': 20,
        'Quantized Size (MB)': 13.3,
        'STFT loss': 0,
        'Mel loss': 0,
    },
    'husein': {
        'Size (MB)': 20,
        'Quantized Size (MB)': 13.3,
        'STFT loss': 0,
        'Mel loss': 0,
    },
}


def available_melgan():
    """
    List available MelGAN Mel-to-Speech models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(_melgan_availability)


def available_mbmelgan():
    """
    List available Multiband MelGAN Mel-to-Speech models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(_melgan_availability)


def melgan(model: str = 'female', quantized = True, **kwargs):
    model = model.lower()
    if model not in _melgan_availability:
        raise Exception(
            'model not supported, please check supported models from `malaya_speech.vocoder.available_melgan()`.'
        )


def mbmelgan(model: str = 'female', quantized = True, **kwargs):
    model = model.lower()
    if model not in _mbmelgan_availability:
        raise Exception(
            'model not supported, please check supported models from `malaya_speech.vocoder.available_mbmelgan()`.'
        )
