from malaya_speech.supervised import stt
from herpetologist import check_type
from malaya_speech.utils import describe_availability
from malaya_speech.stt import _describe

_huggingface_availability = {
    'mesolitica/finetune-whisper-tiny-ms-singlish': {
        'Size (MB)': 151,
        'WER': 0.1322198,
        'CER': 0.0481054,
        'WER-LM': 0.0988016,
        'CER-LM': 0.0411965,
        'Language': ['malay', 'singlish'],
    },
    'mesolitica/finetune-whisper-base-ms-singlish': {
        'Size (MB)': 290,
        'WER': 0.1322198,
        'CER': 0.0481054,
        'WER-LM': 0.0988016,
        'CER-LM': 0.0411965,
        'Language': ['malay', 'singlish'],
    },
    'mesolitica/finetune-whisper-small-ms-singlish': {
        'Size (MB)': 967,
        'WER': 0.1322198,
        'CER': 0.0481054,
        'WER-LM': 0.0988016,
        'CER-LM': 0.0411965,
        'Language': ['malay', 'singlish'],
    },
}


@check_type
def huggingface(
    model: str = 'mesolitica/finetune-whisper-base-ms-singlish',
    force_check: bool = True,
    **kwargs,
):
    """
    Load Finetuned models from HuggingFace.

    Parameters
    ----------
    model : str, optional (default='mesolitica/finetune-whisper-base-ms-singlish')
        Check available models at `malaya_speech.stt.seq2seq.available_huggingface()`.

    Returns
    -------
    result : malaya_speech.model.huggingface.CTC class
    """
    model = model.lower()
    if model not in _huggingface_availability and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.stt.seq2seq.available_huggingface()`.'
        )

    return stt.huggingface_load_seq2seq(model=model, **kwargs)
