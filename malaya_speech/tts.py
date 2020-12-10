_melgan_availability = {
    'male': {'Size (MB)': 20, 'Quantized Size (MB)': 13.3, 'STFT loss': 0},
    'female': {'Size (MB)': 20, 'Quantized Size (MB)': 13.3, 'STFT loss': 0},
}
_hifigan_availability = {
    'male': {'Size (MB)': 20, 'Quantized Size (MB)': 13.3, 'STFT loss': 0},
    'female': {'Size (MB)': 20, 'Quantized Size (MB)': 13.3, 'STFT loss': 0},
}
_tacotron2_availability = {'male': {}, 'female': {}}
_fastspeech2_availability = {'male': {}, 'female': {}}


def available_melgan():
    """
    List available MelGAN Vocoder models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(_melgan_availability)


def available_hifigan():
    """
    List available HifiGAN Vocoder models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(_hifigan_availability)


def melgan(model: str = 'male', quantized: bool = False, **kwargs):
    model = model.lower()
    if model not in _melgan_availability:
        raise Exception(
            'model not supported, please check supported models from `malaya_speech.tts.available_melgan()`.'
        )


def hifigan(model: str = 'male', quantized: bool = False, **kwargs):
    model = model.lower()
    if model not in _hifigan_availability:
        raise Exception(
            'model not supported, please check supported models from `malaya_speech.tts.available_hifigan()`.'
        )


def tacotron2(model: str = 'male', quantized: bool = False, **kwargs):
    model = model.lower()


def fastspeech2(model: str = 'male', quantized: bool = False, **kwargs):
    model = model.lower()
