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

logger = logging.getLogger(__name__)

available_vits = {
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
    'mesolitica/VITS-husein': {
        'Size (MB)': 145,
        'Understand punctuation': True,
        'Is lowercase': False,
        'num speakers': 1,
    },
    'mesolitica/VITS-jordan': {
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
    'mesolitica/VITS-multispeaker-noisy': {
        'Size (MB)': 159,
        'Understand punctuation': True,
        'Is lowercase': False,
        'num speakers': 3,
    },
}

available_vits_v2 = {
    'mesolitica/VITS-V2-husein': {
        'Size (MB)': 145,
        'Understand punctuation': True,
        'Is lowercase': False,
        'num speakers': 1,
    },
}


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

    if version.parse(malaya.__version__) < version.parse('5.1'):
        logger.warning('To get better speech synthesis, make sure Malaya version >= 5.1')

    normalizer = malaya.normalize.normalizer()
    sentence_tokenizer = malaya.text.function.split_into_sentences

    return TextIDS(
        pad_to=pad_to,
        understand_punct=understand_punct,
        is_lower=is_lower,
        normalizer=normalizer,
        sentence_tokenizer=sentence_tokenizer,
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

    if model not in available_vits:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.tts.available_vits()`.'
        )

    selected_model = available_vits[model]

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


def vits_v2(model: str = 'mesolitica/VITS-V2-husein', **kwargs):
    """
    Load VITS V2 End-to-End TTS model.

    Parameters
    ----------
    model : str, optional (default='mesolitica/VITS-V2-husein')
        Check available models at `malaya_speech.tts.available_vits()`.
    Returns
    -------
    result : malaya_speech.torch_model.synthesis.VITS class
    """

    if model not in available_vits_v2:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.tts.available_vits_v2()`.'
        )

    selected_model = available_vits_v2[model]

    text_ids = load_text_ids(
        pad_to=None,
        understand_punct=selected_model['Understand punctuation'],
        is_lower=selected_model['Is lowercase'],
        **kwargs
    )
    return tts.vits_torch_load(
        model=model,
        normalizer=text_ids,
        v2=True,
        **kwargs,
    )
