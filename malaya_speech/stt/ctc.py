from malaya_speech.supervised import stt
from malaya_speech.stt import info

available_huggingface = {
    'mesolitica/conformer-tiny-ctc': {
        'Language': ['malay'],
    },
    'mesolitica/conformer-super-tiny-ctc': {
        'Language': ['malay'],
    },
    'mesolitica/conformer-2x-super-tiny-ctc': {
        'Language': ['malay'],
    },
    'mesolitica/wav2vec2-xls-r-300m-mixed': {
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
    },
    'mesolitica/wav2vec2-xls-r-300m-mixed-v2': {
        'Size (MB)': 1180,
        'malay-malaya': {
            'WER': 0.154782923,
            'CER': 0.035164031,
            'WER-LM': 0.09177104918,
            'CER-LM': 0.0253532288,
        },
        'malay-fleur102': {
            'WER': 0.2013994374,
            'CER': 0.0518170369,
            'WER-LM': 0.14364611216,
            'CER-LM': 0.0416905231,
        },
        'singlish': {
            'WER': 0.2258822139,
            'CER': 0.082982312,
            'WER-LM': 0.17862153528,
            'CER-LM': 0.0718263800,
        },
        'Language': ['malay', 'singlish'],
    },

}


def huggingface(
    model: str = 'mesolitica/wav2vec2-xls-r-300m-mixed',
    force_check: bool = True,
    **kwargs,
):
    """
    Load Finetuned models from HuggingFace.

    Parameters
    ----------
    model : str, optional (default='mesolitica/wav2vec2-xls-r-300m-mixed')
        Check available models at `malaya_speech.stt.ctc.available_huggingface()`.
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
