from malaya_speech.path import (
    PATH_TTS_TACOTRON2,
    S3_PATH_TTS_TACOTRON2,
    PATH_TTS_FASTSPEECH2,
    S3_PATH_TTS_FASTSPEECH2,
    PATH_TTS_FASTPITCH,
    S3_PATH_TTS_FASTPITCH,
    PATH_TTS_GLOWTTS,
    S3_PATH_TTS_GLOWTTS,
)
from malaya_speech.utils.text import (
    convert_to_ascii,
    collapse_whitespace,
    put_spacing_num,
    tts_encode,
    TextIDS,
)
from malaya_speech.supervised import tts
import numpy as np
import re

_tacotron2_availability = {
    'male': {
        'Size (MB)': 104,
        'Quantized Size (MB)': 26.3,
        'Combined loss': 0.1838,
    },
    'female': {
        'Size (MB)': 104,
        'Quantized Size (MB)': 26.3,
        'Combined loss': 0.1887,
    },
    'husein': {
        'Size (MB)': 104,
        'Quantized Size (MB)': 26.3,
        'Combined loss': 0.1165,
    },
    'haqkiem': {
        'Size (MB)': 104,
        'Quantized Size (MB)': 26.3,
        'Combined loss': 0.1375,
    },
    'female-singlish': {
        'Size (MB)': 104,
        'Quantized Size (MB)': 26.3,
        'Combined loss': 0.0923,
    },
}
_fastspeech2_availability = {
    'male': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 31.7,
        'Combined loss': 1.8,
    },
    'female': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 31.7,
        'Combined loss': 1.932,
    },
    'husein': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 31.7,
        'Combined loss': 0.5832,
    },
    'haqkiem': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 31.7,
        'Combined loss': 0.5663,
    },
    'female-singlish': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 31.7,
        'Combined loss': 0.5112,
    },
}

_fastpitch_availability = {
    'male': {
        'Size (MB)': 123,
        'Quantized Size (MB)': 31.1,
        'Combined loss': 1.614,
    },
    'female': {
        'Size (MB)': 123,
        'Quantized Size (MB)': 31.1,
        'Combined loss': 1.669,
    },
    'husein': {
        'Size (MB)': 123,
        'Quantized Size (MB)': 31.1,
        'Combined loss': 0.52515,
    },
    'haqkiem': {
        'Size (MB)': 123,
        'Quantized Size (MB)': 31.1,
        'Combined loss': 0.5186,
    },
}

_glowtts_availability = {
    'male': {
        'Size (MB)': 119,
        'Quantized Size (MB)': 27.6,
        'Combined loss': -1.429,
    },
    'female': {
        'Size (MB)': 119,
        'Quantized Size (MB)': 27.6,
        'Combined loss': -1.464,
    },
    'haqkiem': {
        'Size (MB)': 119,
        'Quantized Size (MB)': 27.6,
        'Combined loss': -1.649,
    },
    'female-singlish': {
        'Size (MB)': 119,
        'Quantized Size (MB)': 27.6,
        'Combined loss': -1.728,
    },
    'multispeaker': {
        'Size (MB)': 404,
        'Quantized Size (MB)': 79.9,
        'Combined loss': -1.882,
    }
}


def load_text_ids(
    pad_to: int = 8,
    understand_punct: bool = True,
    true_case_model=None,
    **kwargs
):
    try:
        import malaya
    except BaseException:
        raise ModuleNotFoundError(
            'malaya not installed. Please install it by `pip install malaya` and try again.'
        )

    normalizer = malaya.normalize.normalizer(date=False, time=False)
    sentence_tokenizer = malaya.text.function.split_into_sentences

    return TextIDS(
        pad_to=pad_to,
        understand_punct=understand_punct,
        normalizer=normalizer,
        sentence_tokenizer=sentence_tokenizer,
        true_case_model=true_case_model,
    )


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


def tacotron2(
    model: str = 'male',
    quantized: bool = False,
    pad_to: int = 8,
    true_case_model=None,
    **kwargs
):
    """
    Load Tacotron2 TTS model.

    Parameters
    ----------
    model : str, optional (default='male')
        Model architecture supported. Allowed values:

        * ``'female'`` - Tacotron2 trained on female voice.
        * ``'male'`` - Tacotron2 trained on male voice.
        * ``'husein'`` - Tacotron2 trained on Husein voice, https://www.linkedin.com/in/husein-zolkepli/
        * ``'haqkiem'`` - Tacotron2 trained on Haqkiem voice, https://www.linkedin.com/in/haqkiem-daim/
        * ``'female-singlish'`` - Tacotron2 trained on female Singlish voice, https://www.imda.gov.sg/programme-listing/digital-services-lab/national-speech-corpus

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.
    pad_to : int, optional (default=8)
        size of pad character with 0. Increase can stable up prediction on short sentence, we trained on 8.
    true_case_model: str, optional (default=None)
        load any true case model, eg, malaya true case model from https://malaya.readthedocs.io/en/latest/load-true-case.html
        the interface must accept a string, return a string, eg, string = true_case_model(string)

    Returns
    -------
    result : malaya_speech.model.tf.Tacotron class
    """
    model = model.lower()

    if model not in _tacotron2_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.tts.available_tacotron2()`.'
        )

    text_ids = load_text_ids(
        pad_to=pad_to,
        true_case_model=true_case_model,
        quantized=quantized,
        **kwargs
    )

    return tts.tacotron_load(
        path=PATH_TTS_TACOTRON2,
        s3_path=S3_PATH_TTS_TACOTRON2,
        model=model,
        name='text-to-speech',
        normalizer=text_ids,
        quantized=quantized,
        **kwargs
    )


def fastspeech2(
    model: str = 'male',
    quantized: bool = False,
    pad_to: int = 8,
    true_case_model=None,
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
        * ``'female-singlish'`` - Fastspeech2 trained on female Singlish voice, https://www.imda.gov.sg/programme-listing/digital-services-lab/national-speech-corpus

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.
    pad_to : int, optional (default=8)
        size of pad character with 0. Increase can stable up prediction on short sentence, we trained on 8.
    true_case_model: str, optional (default=None)
        load any true case model, eg, malaya true case model from https://malaya.readthedocs.io/en/latest/load-true-case.html
        the interface must accept a string, return a string, eg, string = true_case_model(string)

    Returns
    -------
    result : malaya_speech.model.tf.Fastspeech class
    """

    model = model.lower()

    if model not in _fastspeech2_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.tts.available_fastspeech2()`.'
        )

    text_ids = load_text_ids(
        pad_to=pad_to,
        true_case_model=true_case_model,
        quantized=quantized,
        **kwargs
    )
    return tts.fastspeech_load(
        path=PATH_TTS_FASTSPEECH2,
        s3_path=S3_PATH_TTS_FASTSPEECH2,
        model=model,
        name='text-to-speech',
        normalizer=text_ids,
        quantized=quantized,
        **kwargs
    )


def fastpitch(
    model: str = 'male',
    quantized: bool = False,
    pad_to: int = 8,
    true_case_model=None,
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
    true_case_model: str, optional (default=None)
        load any true case model, eg, malaya true case model from https://malaya.readthedocs.io/en/latest/load-true-case.html
        the interface must accept a string, return a string, eg, string = true_case_model(string)

    Returns
    -------
    result : malaya_speech.model.tf.Fastpitch class
    """

    model = model.lower()

    if model not in _fastpitch_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.tts.available_fastpitch()`.'
        )

    text_ids = load_text_ids(
        pad_to=pad_to,
        true_case_model=true_case_model,
        quantized=quantized,
        **kwargs
    )
    return tts.fastpitch_load(
        path=PATH_TTS_FASTPITCH,
        s3_path=S3_PATH_TTS_FASTPITCH,
        model=model,
        name='text-to-speech',
        normalizer=text_ids,
        quantized=quantized,
        **kwargs
    )


def glowtts(model: str = 'male',
            quantized: bool = False,
            pad_to: int = 2,
            true_case_model=None,
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
        * ``'multispeaker'`` - Multispeaker GlowTTS trained on male, female, husein and haqkiem voices, also able to do voice conversion.

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.
    pad_to : int, optional (default=2)
        size of pad character with 0. Increase can stable up prediction on short sentence, we trained on 2.
    true_case_model: str, optional (default=None)
        load any true case model, eg, malaya true case model from https://malaya.readthedocs.io/en/latest/load-true-case.html
        the interface must accept a string, return a string, eg, string = true_case_model(string)

    Returns
    -------
    result : malaya_speech.model.tf.GlowTTS class
    """

    model = model.lower()

    if model not in _glowtts_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.tts.available_glowtts()`.'
        )

    text_ids = load_text_ids(
        pad_to=pad_to,
        true_case_model=true_case_model,
        quantized=quantized,
        **kwargs
    )
    return tts.glowtts_load(
        path=PATH_TTS_GLOWTTS,
        s3_path=S3_PATH_TTS_GLOWTTS,
        model=model,
        name='text-to-speech',
        normalizer=text_ids,
        quantized=quantized,
        **kwargs
    )
