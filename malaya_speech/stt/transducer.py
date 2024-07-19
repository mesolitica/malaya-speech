from malaya_speech.supervised import stt
from malaya_speech.stt import info


available_huggingface = {
    'mesolitica/conformer-tiny': {
        'Size (MB)': 38.5,
        'malay-malaya': {
            'WER': 0.17341180814,
            'CER': 0.059574850240,
        },
        'malay-fleur102': {
            'WER': 0.19524478979,
            'CER': 0.0830808938,
        },
        'Language': ['malay'],
        'Understand punctuation': False,
        'Is lowercase': True,
    },
    'mesolitica/conformer-base': {
        'Size (MB)': 121,
        'malay-malaya': {
            'WER': 0.122076123261,
            'CER': 0.03879606324,
        },
        'malay-fleur102': {
            'WER': 0.1326737206665,
            'CER': 0.050329148570,
        },
        'Language': ['malay'],
        'Understand punctuation': False,
        'Is lowercase': True,
    },
    'mesolitica/conformer-medium': {
        'Size (MB)': 243,
        'malay-malaya': {
            'WER': 0.1054817492564,
            'CER': 0.0313518992842,
        },
        'malay-fleur102': {
            'WER': 0.1172708897486,
            'CER': 0.04310504880,
        },
        'Language': ['malay'],
        'Understand punctuation': False,
        'Is lowercase': True,
    },
    'mesolitica/emformer-base': {
        'Size (MB)': 162,
        'malay-malaya': {
            'WER': 0.175762423786,
            'CER': 0.06233919000537,
        },
        'malay-fleur102': {
            'WER': 0.18303839134,
            'CER': 0.0773853362,
        },
        'Language': ['malay'],
        'Understand punctuation': False,
        'Is lowercase': True,
    },
    'mesolitica/conformer-base-singlish': {
        'Size (MB)': 121,
        'singlish': {
            'WER': 0.06517537334361,
            'CER': 0.03265430876,
        },
        'Language': ['singlish'],
        'Understand punctuation': False,
        'Is lowercase': True,
    },
    'mesolitica/conformer-medium-mixed': {
        'Size (MB)': 243,
        'malay-malaya': {
            'WER': 0.111166517935,
            'CER': 0.03410958328,
        },
        'malay-fleur102': {
            'WER': 0.108354748,
            'CER': 0.037785722,
        },
        'singlish': {
            'WER': 0.091969755225,
            'CER': 0.044627194623,
        },
        'Language': ['malay', 'singlish'],
        'Understand punctuation': False,
        'Is lowercase': True,
    },
    'mesolitica/conformer-medium-malay-whisper': {
        'Size (MB)': 243,
        'malay-malaya': {
            'WER': 0.092561502,
            'CER': 0.0245421736,
        },
        'malay-fleur102': {
            'WER': 0.097128574,
            'CER': 0.03392603,
        },
        'whisper-mixed': {
            'WER': 0.1705298134,
            'CER': 0.10580679153,
        },
        'Language': ['malay', 'mixed'],
        'Understand punctuation': False,
        'Is lowercase': True,
    },
    'mesolitica/conformer-large-malay-whisper': {
        'Size (MB)': 413,
        'malay-malaya': {
            'WER': 0.091311844,
            'CER': 0.02559257576,
        },
        'malay-fleur102': {
            'WER': 0.08824373766,
            'CER': 0.02993509570,
        },
        'whisper-mixed': {
            'WER': 0.1931539167,
            'CER': 0.113297559671,
        },
        'Language': ['malay', 'singlish', 'mixed'],
        'Understand punctuation': False,
        'Is lowercase': True,
    },
}


def huggingface(
    model: str = 'mesolitica/conformer-base',
    **kwargs,
):
    """
    Load Encoder-Transducer ASR model using Pytorch.

    Parameters
    ----------
    model : str, optional (default='mesolitica/conformer-base')
        Check available models at `malaya_speech.stt.transducer.available_huggingface`.

    Returns
    -------
    result : malaya_speech.torch_model.torchaudio.Conformer class
    """

    if model not in available_huggingface:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.stt.transducer.available_huggingface`.'
        )

    return stt.torchaudio(model=model, **kwargs)
