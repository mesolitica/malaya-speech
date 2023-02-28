from malaya_speech.supervised import vocoder
from herpetologist import check_type
from malaya_speech.utils import describe_availability
import warnings

_melgan_availability = {
    'male': {
        'Size (MB)': 17.3,
        'Quantized Size (MB)': 4.53,
    },
    'female': {
        'Size (MB)': 17.3,
        'Quantized Size (MB)': 4.53,
    },
    'husein': {
        'Size (MB)': 17.3,
        'Quantized Size (MB)': 4.53,
    },
    'haqkiem': {
        'Size (MB)': 17.3,
        'Quantized Size (MB)': 4.53,
    },
    'yasmin': {
        'Size (MB)': 17.3,
        'Quantized Size (MB)': 4.53,
    },
    'osman': {
        'Size (MB)': 17.3,
        'Quantized Size (MB)': 4.53,
        'Mel loss': 0.4819,
    },
    'universal': {
        'Size (MB)': 309,
        'Quantized Size (MB)': 77.5,
    },
    'universal-1024': {
        'Size (MB)': 78.4,
        'Quantized Size (MB)': 19.9,
    },
    'universal-384': {
        'Size (MB)': 11.3,
        'Quantized Size (MB)': 3.06,
    },
}

_mbmelgan_availability = {
    'female': {
        'Size (MB)': 10.4,
        'Quantized Size (MB)': 2.82,
    },
    'male': {
        'Size (MB)': 10.4,
        'Quantized Size (MB)': 2.82,
    },
    'husein': {
        'Size (MB)': 10.4,
        'Quantized Size (MB)': 2.82,
    },
    'haqkiem': {
        'Size (MB)': 10.4,
        'Quantized Size (MB)': 2.82,
    },
}

_hifigan_availability = {
    'male': {
        'Size (MB)': 8.8,
        'Quantized Size (MB)': 2.49,
    },
    'female': {
        'Size (MB)': 8.8,
        'Quantized Size (MB)': 2.49,
    },
    'universal-1024': {
        'Size (MB)': 170,
        'Quantized Size (MB)': 42.9,
    },
    'universal-768': {
        'Size (MB)': 72.8,
        'Quantized Size (MB)': 18.5,
    },
    'universal-512': {
        'Size (MB)': 32.6,
        'Quantized Size (MB)': 8.6,
    },
}

_pt_hifigan_availability = {
    'huseinzol05/jik876-UNIVERSAL_V1': {
        'original from': 'https://github.com/jik876/hifi-gan',
        'Size (MB)': 55.8
    }
}


def available_melgan():
    """
    List available MelGAN Mel-to-Speech models.
    """
    warnings.warn(
        '`malaya_speech.vocoder.available_melgan` is using Tensorflow, malaya-speech no longer improved it after version 1.4.0',
        DeprecationWarning)

    return describe_availability(_melgan_availability)


def available_mbmelgan():
    """
    List available Multiband MelGAN Mel-to-Speech models.
    """
    warnings.warn(
        '`malaya_speech.vocoder.available_mbmelgan` is using Tensorflow, malaya-speech no longer improved it after version 1.4.0',
        DeprecationWarning)

    return describe_availability(_mbmelgan_availability)


def available_hifigan():
    """
    List available HiFiGAN Mel-to-Speech models.
    """
    warnings.warn(
        '`malaya_speech.vocoder.available_hifigan` is using Tensorflow, malaya-speech no longer improved it after version 1.4.0',
        DeprecationWarning)

    return describe_availability(_hifigan_availability)


def available_pt_hifigan():
    """
    List available PyTorch HiFiGAN Mel-to-Speech models.
    """
    return describe_availability(_pt_hifigan_availability)


@check_type
def melgan(model: str = 'universal-1024', quantized: bool = False, **kwargs):
    """
    Load MelGAN Vocoder model.

    Parameters
    ----------
    model : str, optional (default='universal-1024')
        Check available models at `malaya_speech.vocoder.available_melgan()`.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.model.synthesis.Vocoder class
    """

    warnings.warn(
        '`malaya_speech.vocoder.melgan` is using Tensorflow, malaya-speech no longer improved it after version 1.4.0',
        DeprecationWarning)

    model = model.lower()
    if model not in _melgan_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.vocoder.available_melgan()`.'
        )

    return vocoder.load(
        model=model,
        module='vocoder-melgan',
        quantized=quantized,
        **kwargs
    )


@check_type
def mbmelgan(model: str = 'female', quantized: bool = False, **kwargs):
    """
    Load Multiband MelGAN Vocoder model.

    Parameters
    ----------
    model : str, optional (default='female')
        Check available models at `malaya_speech.vocoder.available_mbmelgan()`.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.model.synthesis.Vocoder class
    """

    warnings.warn(
        '`malaya_speech.vocoder.mbmelgan` is using Tensorflow, malaya-speech no longer improved it after version 1.4.0',
        DeprecationWarning)

    model = model.lower()
    if model not in _mbmelgan_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.vocoder.available_mbmelgan()`.'
        )
    return vocoder.load(
        model=model,
        module='vocoder-mbmelgan',
        quantized=quantized,
        **kwargs
    )


@check_type
def hifigan(model: str = 'universal-768', quantized: bool = False, **kwargs):
    """
    Load HiFiGAN Vocoder model.

    Parameters
    ----------
    model : str, optional (default='universal-768')
        Check available models at `malaya_speech.vocoder.available_hifigan()`.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.model.synthesis.Vocoder class
    """

    warnings.warn(
        '`malaya_speech.vocoder.hifigan` is using Tensorflow, malaya-speech no longer improved it after version 1.4.0',
        DeprecationWarning)

    model = model.lower()
    if model not in _hifigan_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.vocoder.available_hifigan()`.'
        )
    return vocoder.load(
        model=model,
        module='vocoder-hifigan',
        quantized=quantized,
        **kwargs
    )


def pt_hifigan(model: str = 'huseinzol05/jik876-UNIVERSAL_V1', **kwargs):
    """
    Load PyTorch HiFiGAN Vocoder model, originally from https://github.com/jik876/hifi-gan.

    Parameters
    ----------
    model : str, optional (default='huseinzol05/jik876-UNIVERSAL_V1')

    Returns
    -------
    result : malaya_speech.torch_model.synthesis.Vocoder class
    """

    return vocoder.load_pt_hifigan(
        model=model,
        **kwargs,
    )
