from malaya_speech.supervised import stt
from herpetologist import check_type
from malaya_speech.utils import describe_availability
from malaya_speech.stt import _describe

_transformer_availability = {
    'tiny-conformer': {
        'Size (MB)': 24.4,
        'Quantized Size (MB)': 9.14,
        'malay-malaya': {
            'WER': 0.2128108,
            'CER': 0.08136871,
            'WER-LM': 0.1996828,
            'CER-LM': 0.0770037,
        },
        'malay-fleur102': {
            'WER': 0.2682816,
            'CER': 0.13052725,
            'WER-LM': 0.2579103662,
            'CER-LM': 0.1282339751,
        },
        'Language': ['malay'],
    },
    'small-conformer': {
        'Size (MB)': 49.2,
        'Quantized Size (MB)': 18.1,
        'malay-malaya': {
            'WER': 0.19853302,
            'CER': 0.07449528,
            'WER-LM': 0.18536073,
            'CER-LM': 0.07114307,
        },
        'malay-fleur102': {
            'WER': 0.23412149,
            'CER': 0.1138314813,
            'WER-LM': 0.2214152004,
            'CER-LM': 0.107780986,
        },
        'Language': ['malay'],
    },
    'conformer': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 37.1,
        'malay-malaya': {
            'WER': 0.16340855635999124,
            'CER': 0.058972052118,
            'WER-LM': 0.15618217133,
            'CER-LM': 0.057189785465,
        },
        'malay-fleur102': {
            'WER': 0.20090442596,
            'CER': 0.096169010,
            'WER-LM': 0.1839720400,
            'CER-LM': 0.0881132802,
        },
        'Language': ['malay'],
    },
    'large-conformer': {
        'Size (MB)': 404,
        'Quantized Size (MB)': 107,
        'malay-malaya': {
            'WER': 0.1566839,
            'CER': 0.0619715,
            'WER-LM': 0.1486221,
            'CER-LM': 0.0590102,
        },
        'malay-fleur102': {
            'WER': 0.1711028238,
            'CER': 0.0779535590,
            'WER-LM': 0.16109801581,
            'CER-LM': 0.0721904249,
        },
        'Language': ['malay'],
    },
    'conformer-stack-2mixed': {
        'Size (MB)': 130,
        'Quantized Size (MB)': 38.5,
        'malay-malaya': {
            'WER': 0.1889883954,
            'CER': 0.0726845531,
            'WER-LM': 0.18594507707,
            'CER-LM': 0.0751404434,
        },
        'malay-fleur102': {
            'WER': 0.244836948,
            'CER': 0.117409327,
            'WER-LM': 0.222422157,
            'CER-LM': 0.108059007,
        },
        'singlish': {
            'WER': 0.08535878149,
            'CER': 0.0452357273822,
            'WER-LM': 0.085162938350,
            'CER-LM': 0.044870668858,
        },
        'Language': ['malay', 'singlish'],
    },
    'small-conformer-singlish': {
        'Size (MB)': 49.2,
        'Quantized Size (MB)': 18.1,
        'singlish': {
            'WER': 0.0878310,
            'CER': 0.0456859,
            'WER-LM': 0.08733263,
            'CER-LM': 0.04531657,
        },
        'Language': ['singlish'],
    },
    'conformer-singlish': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 37.1,
        'singlish': {
            'WER': 0.07779246,
            'CER': 0.0403616,
            'WER-LM': 0.07718602,
            'CER-LM': 0.03986952,
        },
        'Language': ['singlish'],
    },
    'large-conformer-singlish': {
        'Size (MB)': 404,
        'Quantized Size (MB)': 107,
        'singlish': {
            'WER': 0.07014733,
            'CER': 0.03587201,
            'WER-LM': 0.06981206,
            'CER-LM': 0.03572307,
        },
        'Language': ['singlish'],
    },
}

_pt_transformer_availability = {
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
    },
    'mesolitica/conformer-base-singlish': {
        'Size (MB)': 121,
        'singlish': {
            'WER': 0.06517537334361,
            'CER': 0.03265430876,
        },
        'Language': ['singlish'],
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
    },
    'mesolitica/conformer-medium-mixed-augmented': {
        'Size (MB)': 243,
        'malay-malaya': {
            'WER': 0.1015719878,
            'CER': 0.03263609230
        },
        'malay-fleur102': {
            'WER': 0.1103884742,
            'CER': 0.0385676182,
        },
        'singlish': {
            'WER': 0.086342166,
            'CER': 0.0413572066,
        },
        'Language': ['malay', 'singlish'],
    },
    'mesolitica/conformer-large-mixed-augmented': {
        'Size (MB)': 413,
        'malay-malaya': {
            'WER': 0.0919852874,
            'CER': 0.026612152,
        },
        'malay-fleur102': {
            'WER': 0.103593636,
            'CER': 0.036611048,
        },
        'singlish': {
            'WER': 0.08727157,
            'CER': 0.04318735972
        },
        'Language': ['malay', 'singlish'],
    },
}


def available_transformer():
    """
    List available Encoder-Transducer ASR models using Tensorflow.
    """

    _describe()
    return describe_availability(_transformer_availability)


def available_pt_transformer():
    """
    List available Encoder-Transducer ASR models using Pytorch.
    """

    _describe()
    return describe_availability(_pt_transformer_availability)


@check_type
def transformer(
    model: str = 'conformer',
    quantized: bool = False,
    **kwargs,
):
    """
    Load Encoder-Transducer ASR model using Tensorflow.

    Parameters
    ----------
    model : str, optional (default='conformer')
        Check available models at `malaya_speech.stt.transducer.available_transformer()`.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.model.transducer.Transducer class
    """

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.stt.transducer.available_transformer()`.'
        )

    return stt.transducer_load(
        model=model,
        module='speech-to-text-transducer',
        languages=_transformer_availability[model]['Language'],
        quantized=quantized,
        **kwargs
    )


def pt_transformer(
    model: str = 'mesolitica/conformer-base',
    **kwargs,
):
    """
    Load Encoder-Transducer ASR model using Pytorch.

    Parameters
    ----------
    model : str, optional (default='mesolitica/conformer-base')
        Check available models at `malaya_speech.stt.transducer.available_pt_transformer()`.

    Returns
    -------
    result : malaya_speech.torch_model.torchaudio.Conformer class
    """

    model = model.lower()
    if model not in _pt_transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.stt.transducer.available_pt_transformer()`.'
        )

    return stt.torchaudio(model=model, **kwargs,)
