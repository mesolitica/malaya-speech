from malaya_speech.supervised import stt
from malaya_speech.stt import info

available_huggingface = {
    'mesolitica/malaysian-whisper-tiny': {
        'Size (MB)': 75.5,
        'Language': ['malay', 'mixed'],
    },
    'mesolitica/malaysian-whisper-base': {
        'Size (MB)': 145,
        'Language': ['malay', 'mixed'],
    },
    'mesolitica/malaysian-whisper-small': {
        'Size (MB)': 484,
        'Language': ['malay', 'mixed'],
    },
    'mesolitica/malaysian-whisper-medium': {
        'Size (MB)': 1530,
        'Language': ['malay', 'mixed'],
    },
}


def huggingface(
    model: str = 'mesolitica/malaysian-whisper-small',
    force_check: bool = True,
    **kwargs,
):
    """
    Load Finetuned models from HuggingFace.

    Parameters
    ----------
    model : str, optional (default='mesolitica/malaysian-whisper-small')
        Check available models at `malaya_speech.stt.seq2seq.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result : malaya_speech.torch_model.huggingface.Seq2Seq class
    """
    if model not in available_huggingface and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.stt.seq2seq.available_huggingface`.'
        )

    return stt.seq2seq(model=model, **kwargs)
