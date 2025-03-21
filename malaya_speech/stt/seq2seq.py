from malaya_speech.supervised import stt
from malaya_speech.stt import info

available_huggingface = {
    'mesolitica/malaysian-whisper-tiny': {
        'Size (MB)': 75.5,
        'malay-malaya': {
            'WER': 0.194655128,
            'CER': 0.04775798,
            'WER-LM': 0.12849904267,
            'CER-LM': 0.0357602212,
        },
        'malay-fleur102': {
            'WER': 0.2373861259,
            'CER': 0.0705547800,
            'WER-LM': 0.17169389543,
            'CER-LM': 0.0591631316,
        },
        'singlish': {
            'WER': 0.127588595,
            'CER': 0.0494924979,
            'WER-LM': 0.096820291,
            'CER-LM': 0.042727603,
        },
        'Language': ['malay', 'mixed'],
        'Understand punctuation': True,
        'Is lowercase': False,
    },
    'mesolitica/malaysian-whisper-base': {
        'Size (MB)': 145,
        'malay-malaya': {
            'WER': 0.194655128,
            'CER': 0.04775798,
            'WER-LM': 0.12849904267,
            'CER-LM': 0.0357602212,
        },
        'malay-fleur102': {
            'WER': 0.2373861259,
            'CER': 0.0705547800,
            'WER-LM': 0.17169389543,
            'CER-LM': 0.0591631316,
        },
        'singlish': {
            'WER': 0.127588595,
            'CER': 0.0494924979,
            'WER-LM': 0.096820291,
            'CER-LM': 0.042727603,
        },
        'Language': ['malay', 'mixed'],
        'Understand punctuation': True,
        'Is lowercase': False,
    },
    'mesolitica/malaysian-whisper-small': {
        'Size (MB)': 484,
        'malay-malaya': {
            'WER': 0.194655128,
            'CER': 0.04775798,
            'WER-LM': 0.12849904267,
            'CER-LM': 0.0357602212,
        },
        'malay-fleur102': {
            'WER': 0.2373861259,
            'CER': 0.0705547800,
            'WER-LM': 0.17169389543,
            'CER-LM': 0.0591631316,
        },
        'singlish': {
            'WER': 0.127588595,
            'CER': 0.0494924979,
            'WER-LM': 0.096820291,
            'CER-LM': 0.042727603,
        },
        'Language': ['malay', 'mixed'],
        'Understand punctuation': True,
        'Is lowercase': False,
    },
    'mesolitica/malaysian-whisper-medium': {
        'Size (MB)': 1530,
        'malay-malaya': {
            'WER': 0.194655128,
            'CER': 0.04775798,
            'WER-LM': 0.12849904267,
            'CER-LM': 0.0357602212,
        },
        'malay-fleur102': {
            'WER': 0.2373861259,
            'CER': 0.0705547800,
            'WER-LM': 0.17169389543,
            'CER-LM': 0.0591631316,
        },
        'singlish': {
            'WER': 0.127588595,
            'CER': 0.0494924979,
            'WER-LM': 0.096820291,
            'CER-LM': 0.042727603,
        },
        'Language': ['malay', 'mixed'],
        'Understand punctuation': True,
        'Is lowercase': False,
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
        Check available models at `malaya_speech.stt.seq2seq.available_huggingface`.
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
