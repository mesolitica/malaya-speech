from malaya_speech.utils.text import (
    convert_to_ascii,
    collapse_whitespace,
    put_spacing_num,
    tts_encode,
    TextIDS,
)
from malaya_speech.supervised import tts
from malaya_speech.utils import describe_availability
import numpy as np
import logging
import warnings
from typing import Callable

logger = logging.getLogger(__name__)

_tacotron2_availability = {
    'male': {
        'Size (MB)': 104,
        'Quantized Size (MB)': 26.3,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'female': {
        'Size (MB)': 104,
        'Quantized Size (MB)': 26.3,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'husein': {
        'Size (MB)': 104,
        'Quantized Size (MB)': 26.3,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'haqkiem': {
        'Size (MB)': 104,
        'Quantized Size (MB)': 26.3,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'female-singlish': {
        'Size (MB)': 104,
        'Quantized Size (MB)': 26.3,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'yasmin': {
        'Size (MB)': 104,
        'Quantized Size (MB)': 26.3,
        'Understand punctuation': True,
        'Is lowercase': False,
    },
    'osman': {
        'Size (MB)': 104,
        'Quantized Size (MB)': 26.3,
        'Understand punctuation': True,
        'Is lowercase': False,
    },
}

_fastspeech2_availability = {
    'male': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 31.7,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'female': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 31.7,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'husein': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 31.7,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'haqkiem': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 31.7,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'female-singlish': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 31.7,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'osman': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 31.7,
        'Understand punctuation': True,
        'Is lowercase': False,
    },
    'yasmin': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 31.7,
        'Understand punctuation': True,
        'Is lowercase': False,
    },
    'yasmin-sdp': {
        'Size (MB)': 128,
        'Quantized Size (MB)': 33.1,
        'Understand punctuation': True,
        'Is lowercase': False,
    },
    'osman-sdp': {
        'Size (MB)': 128,
        'Quantized Size (MB)': 33.1,
        'Understand punctuation': True,
        'Is lowercase': False,
    },
}

_fastpitch_availability = {
    'male': {
        'Size (MB)': 123,
        'Quantized Size (MB)': 31.1,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'female': {
        'Size (MB)': 123,
        'Quantized Size (MB)': 31.1,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'husein': {
        'Size (MB)': 123,
        'Quantized Size (MB)': 31.1,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'haqkiem': {
        'Size (MB)': 123,
        'Quantized Size (MB)': 31.1,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
}

_glowtts_availability = {
    'male': {
        'Size (MB)': 119,
        'Quantized Size (MB)': 27.6,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'female': {
        'Size (MB)': 119,
        'Quantized Size (MB)': 27.6,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'haqkiem': {
        'Size (MB)': 119,
        'Quantized Size (MB)': 27.6,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'female-singlish': {
        'Size (MB)': 119,
        'Quantized Size (MB)': 27.6,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'yasmin': {
        'Size (MB)': 119,
        'Quantized Size (MB)': 27.6,
        'Understand punctuation': True,
        'Is lowercase': False,
    },
    'osman': {
        'Size (MB)': 119,
        'Quantized Size (MB)': 27.6,
        'Understand punctuation': True,
        'Is lowercase': False,
    },
    'multispeaker': {
        'Size (MB)': 404,
        'Quantized Size (MB)': 79.9,
        'Understand punctuation': True,
        'Is lowercase': True,
    }
}


_lightspeech_availability = {
    'yasmin': {
        'Size (MB)': 39.9,
        'Quantized Size (MB)': 10.2,
        'Understand punctuation': True,
        'Is lowercase': False,
    },
    'osman': {
        'Size (MB)': 39.9,
        'Quantized Size (MB)': 10.2,
        'Understand punctuation': True,
        'Is lowercase': False,
    },
}

_vits_availability = {
    'mesolitica/VITS-osman': {
        'Size (MB)': 145,
        'Understand punctuation': True,
        'Is lowercase': False,
        'num speakers': 1,
    },
    'mesolitica/VITS-yasmin': {
        'Size (MB)': 145,
        'Understand punctuation': True,
        'Is lowercase': False,
        'num speakers': 1,
    },
    'mesolitica/VITS-female-singlish': {
        'Size (MB)': 145,
        'Understand punctuation': True,
        'Is lowercase': True,
        'num speakers': 1,
    },
    'mesolitica/VITS-haqkiem': {
        'Size (MB)': 145,
        'Understand punctuation': True,
        'Is lowercase': True,
        'num speakers': 1,
    },
    'mesolitica/VITS-orkid': {
        'Size (MB)': 145,
        'Understand punctuation': True,
        'Is lowercase': False,
        'num speakers': 1,
    },
    'mesolitica/VITS-bunga': {
        'Size (MB)': 145,
        'Understand punctuation': True,
        'Is lowercase': False,
        'num speakers': 1,
    },
    'mesolitica/VITS-jebat': {
        'Size (MB)': 145,
        'Understand punctuation': True,
        'Is lowercase': False,
        'num speakers': 1,
    },
    'mesolitica/VITS-tuah': {
        'Size (MB)': 145,
        'Understand punctuation': True,
        'Is lowercase': False,
        'num speakers': 1,
    },
    'mesolitica/VITS-male': {
        'Size (MB)': 145,
        'Understand punctuation': True,
        'Is lowercase': False,
        'num speakers': 1,
    },
    'mesolitica/VITS-female': {
        'Size (MB)': 145,
        'Understand punctuation': True,
        'Is lowercase': False,
        'num speakers': 1,
    },
    'mesolitica/VITS-multispeaker-clean': {
        'Size (MB)': 159,
        'Understand punctuation': True,
        'Is lowercase': False,
        'num speakers': 9,
    },
}

_e2e_fastspeech2_availability = {
    'osman': {
        'Size (MB)': 167,
        'Quantized Size (MB)': 43.3,
        'Understand punctuation': True,
        'Is lowercase': False,
    },
    'yasmin': {
        'Size (MB)': 167,
        'Quantized Size (MB)': 43.3,
        'Understand punctuation': True,
        'Is lowercase': False,
    },
}


def available_tacotron2():
    """
    List available Tacotron2, Text to Mel models.
    """
    warnings.warn(
        '`malaya_speech.tts.tacotron2` is using Tensorflow, malaya-speech no longer improved it after version 1.4.0',
        DeprecationWarning)

    return describe_availability(_tacotron2_availability)


def available_fastspeech2():
    """
    List available FastSpeech2, Text to Mel models.
    """
    warnings.warn(
        '`malaya_speech.tts.fastspeech2` is using Tensorflow, malaya-speech no longer improved it after version 1.4.0',
        DeprecationWarning)

    return describe_availability(_fastspeech2_availability)


def available_fastpitch():
    """
    List available FastPitch, Text to Mel models.
    """
    warnings.warn(
        '`malaya_speech.tts.fastpitch` is using Tensorflow, malaya-speech no longer improved it after version 1.4.0',
        DeprecationWarning)

    return describe_availability(_fastpitch_availability)


def available_glowtts():
    """
    List available GlowTTS, Text to Mel models.
    """
    warnings.warn(
        '`malaya_speech.tts.glowtts` is using Tensorflow, malaya-speech no longer improved it after version 1.4.0',
        DeprecationWarning)

    return describe_availability(_glowtts_availability)


def available_lightspeech():
    """
    List available LightSpeech, Text to Mel models.
    """
    warnings.warn(
        '`malaya_speech.tts.lightspeech` is using Tensorflow, malaya-speech no longer improved it after version 1.4.0',
        DeprecationWarning)

    return describe_availability(_e2e_fastspeech2_availability)


def available_e2e_fastspeech2():
    """
    List available FastSpeech2, End-to-End models.
    """
    warnings.warn(
        '`malaya_speech.tts.e2e_fastspeech2` is using Tensorflow, malaya-speech no longer improved it after version 1.4.0',
        DeprecationWarning)

    return describe_availability(_e2e_fastspeech2_availability)


def available_vits():
    """
    List available VITS, End-to-End models.
    """

    return describe_availability(_vits_availability)


def load_text_ids(
    pad_to: int = 8,
    understand_punct: bool = True,
    is_lower: bool = True,
    **kwargs,
):
    """
    Load text normalizer module use by Malaya-Speech TTS.
    """

    try:
        import malaya
        from packaging import version
    except BaseException:
        raise ModuleNotFoundError(
            'malaya not installed. Please install it by `pip install malaya` and try again.'
        )

    if version.parse(malaya.__version__) < version.parse('4.9.1'):
        logger.warning('To get better speech synthesis, make sure Malaya version >= 4.9.1')

    normalizer = malaya.normalize.normalizer()
    sentence_tokenizer = malaya.text.function.split_into_sentences

    return TextIDS(
        pad_to=pad_to,
        understand_punct=understand_punct,
        is_lower=is_lower,
        normalizer=normalizer,
        sentence_tokenizer=sentence_tokenizer,
    )


def tacotron2(
    model: str = 'yasmin',
    quantized: bool = False,
    pad_to: int = 8,
    **kwargs,
):
    """
    Load Tacotron2 Text-to-Mel TTS model.

    Parameters
    ----------
    model : str, optional (default='yasmin')
        Check available models at `malaya_speech.tts.available_tacotron2()`.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.
    pad_to : int, optional (default=8)
        size of pad character with 0. Increase can stable up prediction on short sentence, we trained on 8.

    Returns
    -------
    result : malaya_speech.model.synthesis.Tacotron class
    """
    warnings.warn(
        '`malaya_speech.tts.tacotron2` is using Tensorflow, malaya-speech no longer improved it after version 1.4.0',
        DeprecationWarning)

    model = model.lower()

    if model not in _tacotron2_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.tts.available_tacotron2()`.'
        )

    selected_model = _tacotron2_availability[model]

    text_ids = load_text_ids(
        pad_to=pad_to,
        understand_punct=selected_model['Understand punctuation'],
        is_lower=selected_model['Is lowercase'],
        quantized=quantized,
        **kwargs
    )

    return tts.tacotron_load(
        model=model,
        module='text-to-speech-tacotron',
        normalizer=text_ids,
        quantized=quantized,
        **kwargs
    )


def fastspeech2(
    model: str = 'osman',
    quantized: bool = False,
    pad_to: int = 8,
    **kwargs,
):
    """
    Load Fastspeech2 Text-to-Mel TTS model.

    Parameters
    ----------
    model : str, optional (default='male')
        Check available models at `malaya_speech.tts.available_fastspeech2()`.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.
    pad_to : int, optional (default=8)
        size of pad character with 0. Increase can stable up prediction on short sentence, we trained on 8.

    Returns
    -------
    result : malaya_speech.model.synthesis.Fastspeech class
    """

    warnings.warn(
        '`malaya_speech.tts.fastspeech2` is using Tensorflow, malaya-speech no longer improved it after version 1.4.0',
        DeprecationWarning)

    model = model.lower()

    if model not in _fastspeech2_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.tts.available_fastspeech2()`.'
        )

    selected_model = _fastspeech2_availability[model]

    text_ids = load_text_ids(
        pad_to=pad_to,
        understand_punct=selected_model['Understand punctuation'],
        is_lower=selected_model['Is lowercase'],
        quantized=quantized,
        **kwargs
    )
    return tts.fastspeech_load(
        model=model,
        module='text-to-speech-fastspeech',
        normalizer=text_ids,
        quantized=quantized,
        **kwargs
    )


def fastpitch(
    model: str = 'male',
    quantized: bool = False,
    pad_to: int = 8,
    **kwargs,
):
    """
    Load Fastspitch Text-to-Mel TTS model.

    Parameters
    ----------
    model : str, optional (default='male')
        Check available models at `malaya_speech.tts.available_fastpitch()`.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.
    pad_to : int, optional (default=8)
        size of pad character with 0. Increase can stable up prediction on short sentence, we trained on 8.

    Returns
    -------
    result : malaya_speech.model.synthesis.Fastpitch class
    """
    warnings.warn(
        '`malaya_speech.tts.fastpitch` is using Tensorflow, malaya-speech no longer improved it after version 1.4.0',
        DeprecationWarning)

    model = model.lower()

    if model not in _fastpitch_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.tts.available_fastpitch()`.'
        )

    selected_model = _fastpitch_availability[model]

    text_ids = load_text_ids(
        pad_to=pad_to,
        understand_punct=selected_model['Understand punctuation'],
        is_lower=selected_model['Is lowercase'],
        quantized=quantized,
        **kwargs
    )
    return tts.fastpitch_load(
        model=model,
        module='text-to-speech-fastpitch',
        normalizer=text_ids,
        quantized=quantized,
        **kwargs
    )


def glowtts(
    model: str = 'yasmin',
    quantized: bool = False,
    pad_to: int = 2,
    **kwargs,
):
    """
    Load GlowTTS Text-to-Mel TTS model.

    Parameters
    ----------
    model : str, optional (default='yasmin')
        Check available models at `malaya_speech.tts.available_glowtts()`.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.
    pad_to : int, optional (default=2)
        size of pad character with 0. Increase can stable up prediction on short sentence, we trained on 2.

    Returns
    -------
    result : malaya_speech.model.synthesis.GlowTTS class
    """

    warnings.warn(
        '`malaya_speech.tts.glowtts` is using Tensorflow, malaya-speech no longer improved it after version 1.4.0',
        DeprecationWarning)

    model = model.lower()

    if model not in _glowtts_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.tts.available_glowtts()`.'
        )

    selected_model = _glowtts_availability[model]

    text_ids = load_text_ids(
        pad_to=pad_to,
        understand_punct=selected_model['Understand punctuation'],
        is_lower=selected_model['Is lowercase'],
        quantized=quantized,
        **kwargs
    )

    return tts.glowtts_load(
        model=model,
        module='text-to-speech-glowtts',
        normalizer=text_ids,
        quantized=quantized,
        **kwargs
    )


def lightspeech(
    model: str = 'male',
    quantized: bool = False,
    pad_to: int = 8,
    **kwargs,
):
    """
    Load LightSpeech Text-to-Mel TTS model.

    Parameters
    ----------
    model : str, optional (default='male')
        Check available models at `malaya_speech.tts.available_lightspeech()`.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.
    pad_to : int, optional (default=8)
        size of pad character with 0. Increase can stable up prediction on short sentence, we trained on 8.

    Returns
    -------
    result : malaya_speech.model.synthesis.Fastspeech class
    """
    warnings.warn(
        '`malaya_speech.tts.lightspeech` is using Tensorflow, malaya-speech no longer improved it after version 1.4.0',
        DeprecationWarning)

    model = model.lower()

    if model not in _lightspeech_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.tts.available_lightspeech()`.'
        )

    selected_model = _lightspeech_availability[model]

    text_ids = load_text_ids(
        pad_to=pad_to,
        understand_punct=selected_model['Understand punctuation'],
        is_lower=selected_model['Is lowercase'],
        quantized=quantized,
        **kwargs
    )
    return tts.fastspeech_load(
        model=model,
        module='text-to-speech-lightspeech',
        normalizer=text_ids,
        quantized=quantized,
        **kwargs
    )


def e2e_fastspeech2(
    model: str = 'osman',
    quantized: bool = False,
    pad_to: int = 8,
    **kwargs,
):
    """
    Load Fastspeech2 Text-to-Mel TTS model.

    Parameters
    ----------
    model : str, optional (default='male')
        Check available models at `malaya_speech.tts.available_e2e_fastspeech2()`.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.
    pad_to : int, optional (default=8)
        size of pad character with 0. Increase can stable up prediction on short sentence, we trained on 8.

    Returns
    -------
    result : malaya_speech.model.synthesis.E2E_FastSpeech class
    """
    warnings.warn(
        '`malaya_speech.tts.e2e_fastspeech2` is using Tensorflow, malaya-speech no longer improved it after version 1.4.0',
        DeprecationWarning)

    model = model.lower()
    if model not in _e2e_fastspeech2_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.tts.available_e2e_fastspeech2()`.'
        )

    selected_model = _e2e_fastspeech2_availability[model]

    text_ids = load_text_ids(
        pad_to=pad_to,
        understand_punct=selected_model['Understand punctuation'],
        is_lower=selected_model['Is lowercase'],
        quantized=quantized,
        **kwargs
    )
    return tts.e2e_fastspeech_load(
        model=model,
        module='text-to-speech-e2e-fastspeech',
        normalizer=text_ids,
        quantized=quantized,
        **kwargs
    )


def vits(model: str = 'mesolitica/VITS-osman', **kwargs):
    """
    Load VITS End-to-End TTS model.

    Parameters
    ----------
    model : str, optional (default='mesolitica/VITS-osman')
        Check available models at `malaya_speech.tts.available_vits()`.
    Returns
    -------
    result : malaya_speech.torch_model.synthesis.VITS class
    """

    if model not in _vits_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.tts.available_vits()`.'
        )

    selected_model = _vits_availability[model]

    text_ids = load_text_ids(
        pad_to=None,
        understand_punct=selected_model['Understand punctuation'],
        is_lower=selected_model['Is lowercase'],
        **kwargs
    )
    return tts.vits_torch_load(
        model=model,
        normalizer=text_ids,
        **kwargs,
    )
