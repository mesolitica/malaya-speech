from malaya_speech.supervised import stt
from malaya_speech.stt import info

available_huggingface = {
    'mesolitica/conformer-tiny-ctc': {
        'Size (MB)': 15.8,
        'Language': ['malay', 'mixed'],
        'Understand punctuation': False,
        'Is lowercase': True,
    },
    'mesolitica/conformer-super-tiny-ctc': {
        'Size (MB)': 15.8,
        'Language': ['malay', 'mixed'],
        'Understand punctuation': False,
        'Is lowercase': True,
    },
    'mesolitica/w2v-bert-2.0-mixed': {
        'Size (MB)': 1180,
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
        'Language': ['malay', 'singlish'],
        'Understand punctuation': False,
        'Is lowercase': True,
    },
}


def huggingface(
    model: str = 'mesolitica/w2v-bert-2.0-mixed',
    force_check: bool = True,
    **kwargs,
):
    """
    Load Finetuned models from HuggingFace.

    Parameters
    ----------
    model : str, optional (default='mesolitica/w2v-bert-2.0-mixed')
        Check available models at `malaya_speech.stt.ctc.available_huggingface`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result : malaya_speech.torch_model.huggingface.CTC class
    """
    if model not in available_huggingface and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.stt.ctc.available_huggingface`.'
        )

    return stt.ctc(model=model, **kwargs)
