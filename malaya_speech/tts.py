from malaya_speech.utils.text import (
    convert_to_ascii,
    collapse_whitespace,
    put_spacing_num,
    tts_encode,
    TextIDS,
)
from malaya_speech.supervised import tts
import numpy as np
import logging
from typing import Callable

logger = logging.getLogger('malaya_speech.tts')

_tacotron2_availability = {
    'male': {
        'Size (MB)': 104,
        'Quantized Size (MB)': 26.3,
        'Combined loss': 0.1838,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'female': {
        'Size (MB)': 104,
        'Quantized Size (MB)': 26.3,
        'Combined loss': 0.1887,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'husein': {
        'Size (MB)': 104,
        'Quantized Size (MB)': 26.3,
        'Combined loss': 0.1165,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'haqkiem': {
        'Size (MB)': 104,
        'Quantized Size (MB)': 26.3,
        'Combined loss': 0.1375,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'female-singlish': {
        'Size (MB)': 104,
        'Quantized Size (MB)': 26.3,
        'Combined loss': 0.0923,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'yasmin': {
        'Size (MB)': 104,
        'Quantized Size (MB)': 26.3,
        'Combined loss': 0.06874,
        'Understand punctuation': True,
        'Is lowercase': False,
    },
    'osman': {
        'Size (MB)': 104,
        'Quantized Size (MB)': 26.3,
        'Combined loss': 0.06911,
        'Understand punctuation': True,
        'Is lowercase': False,
    },
}

_fastspeech2_availability = {
    'male': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 31.7,
        'Combined loss': 1.8,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'female': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 31.7,
        'Combined loss': 1.932,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'husein': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 31.7,
        'Combined loss': 0.5832,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'haqkiem': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 31.7,
        'Combined loss': 0.5663,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'female-singlish': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 31.7,
        'Combined loss': 0.5112,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'yasmin': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 31.7,
        'Combined loss': 0.7212,
        'Understand punctuation': True,
        'Is lowercase': False,
    },
    'yasmin-small': {
        'Size (MB)': 32.9,
        'Quantized Size (MB)': 8.5,
        'Combined loss': 0.7994,
        'Understand punctuation': True,
        'Is lowercase': False,
    },
    'osman': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 31.7,
        'Combined loss': 0.7341,
        'Understand punctuation': True,
        'Is lowercase': False,
    },
    'osman-small': {
        'Size (MB)': 32.9,
        'Quantized Size (MB)': 8.5,
        'Combined loss': 0.8182,
        'Understand punctuation': True,
        'Is lowercase': False,
    },
}

_fastpitch_availability = {
    'male': {
        'Size (MB)': 123,
        'Quantized Size (MB)': 31.1,
        'Combined loss': 1.614,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'female': {
        'Size (MB)': 123,
        'Quantized Size (MB)': 31.1,
        'Combined loss': 1.669,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'husein': {
        'Size (MB)': 123,
        'Quantized Size (MB)': 31.1,
        'Combined loss': 0.52515,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'haqkiem': {
        'Size (MB)': 123,
        'Quantized Size (MB)': 31.1,
        'Combined loss': 0.5186,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
}

_glowtts_availability = {
    'male': {
        'Size (MB)': 119,
        'Quantized Size (MB)': 27.6,
        'Combined loss': -1.429,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'female': {
        'Size (MB)': 119,
        'Quantized Size (MB)': 27.6,
        'Combined loss': -1.464,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'haqkiem': {
        'Size (MB)': 119,
        'Quantized Size (MB)': 27.6,
        'Combined loss': -1.649,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'female-singlish': {
        'Size (MB)': 119,
        'Quantized Size (MB)': 27.6,
        'Combined loss': -1.728,
        'Understand punctuation': True,
        'Is lowercase': True,
    },
    'yasmin': {
        'Size (MB)': 119,
        'Quantized Size (MB)': 27.6,
        'Combined loss': -1.908,
        'Understand punctuation': True,
        'Is lowercase': False,
    },
    'osman': {
        'Size (MB)': 119,
        'Quantized Size (MB)': 27.6,
        'Combined loss': -1.908,
        'Understand punctuation': True,
        'Is lowercase': False,
    },
    'multispeaker': {
        'Size (MB)': 404,
        'Quantized Size (MB)': 79.9,
        'Combined loss': -1.882,
        'Understand punctuation': True,
        'Is lowercase': True,
    }
}


def available_tacotron2():
    """
    List available Tacotron2, Text to Mel models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(
        _tacotron2_availability,
        text='`husein`, `haqkiem` and `female-singlish` combined loss from training set',
    )


def available_fastspeech2():
    """
    List available FastSpeech2, Text to Mel models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(
        _fastspeech2_availability,
        text='`husein`, `haqkiem` and `female-singlish` combined loss from training set',
    )


def available_fastpitch():
    """
    List available FastPitch, Text to Mel models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(
        _fastpitch_availability,
        text='`husein` and `haqkiem` combined loss from training set',
    )


def available_glowtts():
    """
    List available GlowTTS, Text to Mel models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(
        _glowtts_availability,
        text='`haqkiem` and `female-singlish` combined loss from training set',
    )


def load_text_ids(
    pad_to: int = 8,
    understand_punct: bool = True,
    is_lower: bool = True,
    true_case_model: Callable = None,
    **kwargs
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

    if version.parse(malaya.__version__) < version.parse('4.7.5'):
        logger.warning('To get better speech synthesis, make sure Malaya version >= 4.7.5')

    normalizer = malaya.normalize.normalizer()
    sentence_tokenizer = malaya.text.function.split_into_sentences

    return TextIDS(
        pad_to=pad_to,
        understand_punct=understand_punct,
        is_lower=is_lower,
        normalizer=normalizer,
        sentence_tokenizer=sentence_tokenizer,
        true_case_model=true_case_model,
    )


def tacotron2(
    model: str = 'yasmin',
    quantized: bool = False,
    pad_to: int = 8,
    true_case_model: Callable = None,
    **kwargs
):
    """
    Load Tacotron2 TTS model.

    Parameters
    ----------
    model : str, optional (default='yasmin')
        Model architecture supported. Allowed values:

        * ``'female'`` - Tacotron2 trained on female voice.
        * ``'male'`` - Tacotron2 trained on male voice.
        * ``'husein'`` - Tacotron2 trained on Husein voice, https://www.linkedin.com/in/husein-zolkepli/
        * ``'haqkiem'`` - Tacotron2 trained on Haqkiem voice, https://www.linkedin.com/in/haqkiem-daim/
        * ``'yasmin'`` - Tacotron2 trained on female Yasmin voice.
        * ``'osman'`` - Tacotron2 trained on male Osman voice.
        * ``'female-singlish'`` - Tacotron2 trained on female Singlish voice, https://www.imda.gov.sg/programme-listing/digital-services-lab/national-speech-corpus

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.
    pad_to : int, optional (default=8)
        size of pad character with 0. Increase can stable up prediction on short sentence, we trained on 8.
    true_case_model: Callable, optional (default=None)
        load any true case model, eg, malaya true case model from https://malaya.readthedocs.io/en/latest/load-true-case.html
        the interface must accept a string, return a string, eg, string = true_case_model(string)

    Returns
    -------
    result : malaya_speech.model.synthesis.Tacotron class
    """
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
        true_case_model=true_case_model,
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
    model: str = 'male',
    quantized: bool = False,
    pad_to: int = 8,
    true_case_model: Callable = None,
    **kwargs
):
    """
    Load Fastspeech2 TTS model.

    Parameters
    ----------
    model : str, optional (default='male')
        Model architecture supported. Allowed values:

        * ``'female'`` - Fastspeech2 trained on female voice.
        * ``'male'`` - Fastspeech2 trained on male voice.
        * ``'husein'`` - Fastspeech2 trained on Husein voice, https://www.linkedin.com/in/husein-zolkepli/
        * ``'haqkiem'`` - Fastspeech2 trained on Haqkiem voice, https://www.linkedin.com/in/haqkiem-daim/
        * ``'yasmin'`` - Fastspeech2 trained on female Yasmin voice.
        * ``'osman'`` - Fastspeech2 trained on male Osman voice.
        * ``'female-singlish'`` - Fastspeech2 trained on female Singlish voice, https://www.imda.gov.sg/programme-listing/digital-services-lab/national-speech-corpus

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.
    pad_to : int, optional (default=8)
        size of pad character with 0. Increase can stable up prediction on short sentence, we trained on 8.
    true_case_model: Callable, optional (default=None)
        load any true case model, eg, malaya true case model from https://malaya.readthedocs.io/en/latest/load-true-case.html
        the interface must accept a string, return a string, eg, string = true_case_model(string)

    Returns
    -------
    result : malaya_speech.model.synthesis.Fastspeech class
    """

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
        true_case_model=true_case_model,
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
    true_case_model: Callable = None,
    **kwargs
):
    """
    Load Fastspitch TTS model.

    Parameters
    ----------
    model : str, optional (default='male')
        Model architecture supported. Allowed values:

        * ``'female'`` - Fastpitch trained on female voice.
        * ``'male'`` - Fastpitch trained on male voice.
        * ``'husein'`` - Fastpitch trained on Husein voice, https://www.linkedin.com/in/husein-zolkepli/
        * ``'haqkiem'`` - Fastpitch trained on Haqkiem voice, https://www.linkedin.com/in/haqkiem-daim/

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.
    pad_to : int, optional (default=8)
        size of pad character with 0. Increase can stable up prediction on short sentence, we trained on 8.
    true_case_model: Callable, optional (default=None)
        load any true case model, eg, malaya true case model from https://malaya.readthedocs.io/en/latest/load-true-case.html
        the interface must accept a string, return a string, eg, string = true_case_model(string)

    Returns
    -------
    result : malaya_speech.model.synthesis.Fastpitch class
    """

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
        true_case_model=true_case_model,
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


def glowtts(model: str = 'male',
            quantized: bool = False,
            pad_to: int = 2,
            true_case_model: Callable = None,
            **kwargs):
    """
    Load GlowTTS TTS model.

    Parameters
    ----------
    model : str, optional (default='male')
        Model architecture supported. Allowed values:

        * ``'female'`` - GlowTTS trained on female voice.
        * ``'male'`` - GlowTTS trained on male voice.
        * ``'haqkiem'`` - GlowTTS trained on Haqkiem voice, https://www.linkedin.com/in/haqkiem-daim/
        * ``'female-singlish'`` - GlowTTS trained on female Singlish voice, https://www.imda.gov.sg/programme-listing/digital-services-lab/national-speech-corpus
        * ``'yasmin'`` - GlowTTS trained on female Yasmin voice.
        * ``'osman'`` - GlowTTS trained on male Osman voice.
        * ``'multispeaker'`` - Multispeaker GlowTTS trained on male, female, husein and haqkiem voices, also able to do voice conversion.

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.
    pad_to : int, optional (default=2)
        size of pad character with 0. Increase can stable up prediction on short sentence, we trained on 2.
    true_case_model: Callable, optional (default=None)
        load any true case model, eg, malaya true case model from https://malaya.readthedocs.io/en/latest/load-true-case.html
        the interface must accept a string, return a string, eg, string = true_case_model(string)

    Returns
    -------
    result : malaya_speech.model.synthesis.GlowTTS class
    """

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
        true_case_model=true_case_model,
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
