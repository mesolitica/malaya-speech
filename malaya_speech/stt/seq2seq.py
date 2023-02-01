from malaya_speech.supervised import stt
from herpetologist import check_type
from malaya_speech.utils import describe_availability
from malaya_speech.stt import _describe

_huggingface_availability = {
    'mesolitica/finetune-whisper-tiny-ms-singlish': {
        'Size (MB)': 151,
        'malay-malaya': {
            'WER': 0.20141585,
            'CER': 0.071964908,
        },
        'malay-fleur102': {
            'WER': 0.235680975,
            'CER': 0.0986880877,
        },
        'singlish': {
            'WER': 0.09045121,
            'CER': 0.0481965,
        },
        'Language': ['malay', 'singlish'],
    },
    'mesolitica/finetune-whisper-tiny-ms-singlish-v2': {
        'Size (MB)': 151,
        'malay-malaya': {
            'WER': 0.20141585,
            'CER': 0.071964908,
        },
        'malay-fleur102': {
            'WER': 0.22459602,
            'CER': 0.089406469,
        },
        'singlish': {
            'WER': 0.138882971,
            'CER': 0.074929807,
        },
        'Language': ['malay', 'singlish'],
    },
    'mesolitica/finetune-whisper-base-ms-singlish-v2': {
        'Size (MB)': 290,
        'malay-malaya': {
            'WER': 0.172632664,
            'CER': 0.0680027682,
        },
        'malay-fleur102': {
            'WER': 0.1837319118,
            'CER': 0.0599804251,
        },
        'singlish': {
            'WER': 0.111506313,
            'CER': 0.05852830724,
        },
        'Language': ['malay', 'singlish'],
    },
    'mesolitica/finetune-whisper-small-ms-singlish-v2': {
        'Size (MB)': 967,
        'malay-malaya': {
            'WER': 0.13189875561,
            'CER': 0.0434602169,
        },
        'malay-fleur102': {
            'WER': 0.13277694,
            'CER': 0.0478108612,
        },
        'singlish': {
            'WER': 0.09489335668,
            'CER': 0.05045327551,
        },
        'Language': ['malay', 'singlish'],
    },
}


def available_huggingface():
    """
    List available HuggingFace Seq2Seq ASR models.
    """

    _describe()
    return describe_availability(_huggingface_availability)


def available_whisper():
    """
    List available OpenAI Whisper ASR models.
    """

    _describe()
    return describe_availability(_huggingface_availability)


@check_type
def huggingface(
    model: str = 'mesolitica/finetune-whisper-base-ms-singlish-v2',
    force_check: bool = True,
    **kwargs,
):
    """
    Load Finetuned models from HuggingFace.

    Parameters
    ----------
    model : str, optional (default='mesolitica/finetune-whisper-base-ms-singlish-v2')
        Check available models at `malaya_speech.stt.seq2seq.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result : malaya_speech.model.huggingface.Seq2Seq class
    """
    model = model.lower()
    if model not in _huggingface_availability and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.stt.seq2seq.available_huggingface()`.'
        )

    return stt.huggingface_load_seq2seq(model=model, **kwargs)


def whisper(
    model: str = 'mesolitica/finetune-whisper-base-ms-singlish-v2',
    force_check: bool = True,
    **kwargs,
):
    """
    Load Finetuned models from HuggingFace.

    Parameters
    ----------
    model : str, optional (default='mesolitica/finetune-whisper-base-ms-singlish-v2')
        Check available models at `malaya_speech.stt.seq2seq.available_whisper()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result : whisper.model.Whisper class
    """
    model = model.lower()
    if model not in _huggingface_availability and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.stt.seq2seq.available_whisper()`.'
        )

    return stt.whisper(model=model, **kwargs)
