from malaya_speech.supervised import stt
from herpetologist import check_type
from malaya_speech.utils import describe_availability
from malaya_speech.stt import _describe
import warnings

_transformer_availability = {
    'tiny-conformer': {
        'Size (MB)': 24.4,
        'Quantized Size (MB)': 9.14,
        'WER': 0.25205879818082805,
        'CER': 0.11939432658062442,
        'WER-LM': 0.24466923675904093,
        'CER-LM': 0.12099917108722165,
        'Language': ['malay'],
    },
    'small-conformer': {
        'Size (MB)': 49.2,
        'Quantized Size (MB)': 18.1,
        'WER': 0.2216678092407059,
        'CER': 0.10721028207884116,
        'WER-LM': 0.215915398935267,
        'CER-LM': 0.10903240184809346,
        'Language': ['malay'],
    },
    'conformer': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 37.1,
        'WER': 0.1854134610718644,
        'CER': 0.08612027714394176,
        'WER-LM': 0.18542071537532542,
        'CER-LM': 0.09424689146824226,
        'Language': ['malay'],
    },
    'large-conformer': {
        'Size (MB)': 404,
        'Quantized Size (MB)': 107,
        'WER': 0.16068731055857535,
        'CER': 0.07225488182147317,
        'WER-LM': 0.15852097352196282,
        'CER-LM': 0.07445114519781025,
        'Language': ['malay'],
    },
    'conformer-stack-2mixed': {
        'Size (MB)': 130,
        'Quantized Size (MB)': 38.5,
        'WER': 0.1036084,
        'CER': 0.050069,
        'WER-LM': 0.1029111,
        'CER-LM': 0.0502013,
        'Language': ['malay', 'singlish'],
    },
    'conformer-stack-3mixed': {
        'Size (MB)': 130,
        'Quantized Size (MB)': 38.5,
        'WER': 0.2347684,
        'CER': 0.133944,
        'WER-LM': 0.229241,
        'CER-LM': 0.1307018,
        'Language': ['malay', 'singlish', 'mandarin'],
    },
    'small-conformer-singlish': {
        'Size (MB)': 49.2,
        'Quantized Size (MB)': 18.1,
        'WER': 0.0878310,
        'CER': 0.0456859,
        'WER-LM': 0.08733263,
        'CER-LM': 0.04531657,
        'Language': ['singlish'],
    },
    'conformer-singlish': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 37.1,
        'WER': 0.07779246,
        'CER': 0.0403616,
        'WER-LM': 0.07718602,
        'CER-LM': 0.03986952,
        'Language': ['singlish'],
    },
    'large-conformer-singlish': {
        'Size (MB)': 404,
        'Quantized Size (MB)': 107,
        'WER': 0.07014733,
        'CER': 0.03587201,
        'WER-LM': 0.06981206,
        'CER-LM': 0.03572307,
        'Language': ['singlish'],
    },
    'xs-squeezeformer': {
        'Size (MB)': 51.9,
        'Quantized Size (MB)': 23.4,
        'WER': 0.24361950288615478,
        'CER': 0.1101217363425894,
        'WER-LM': 0.23732601782189705,
        'CER-LM': 0.11326849769270286,
        'Language': ['malay'],
    },
    'sm-squeezeformer': {
        'Size (MB)': 147,
        'Quantized Size (MB)': 47.4,
        'WER': 0.1995528388961415,
        'CER': 0.09630517271889948,
        'WER-LM': 0.1942312047785515,
        'CER-LM': 0.09945312677604806,
        'Language': ['malay'],
    },
    'm-squeezeformer': {
        'Size (MB)': 261,
        'Quantized Size (MB)': 78.5,
        'WER': 0.19985013387051873,
        'CER': 0.09779469719228841,
        'WER-LM': 0.19800923596045886,
        'CER-LM': 0.10461433715575515,
        'Language': ['malay'],
    },
}


def available_transformer():
    """
    List available Encoder-Transducer ASR models.
    """

    _describe()
    return describe_availability(_transformer_availability)


@check_type
def transformer(
    model: str = 'conformer',
    quantized: bool = False,
    **kwargs,
):
    """
    Load Encoder-Transducer ASR model.

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

    warnings.warn(
        '`malaya.stt.transducer.transformer` is using Tensorflow, means malaya-speech no longer improved it.', DeprecationWarning)

    model = model.lower()
    if model not in _transducer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.stt.transducer.available_transformer()`.'
        )

    return stt.transducer_load(
        model=model,
        module='speech-to-text-transducer',
        languages=_transducer_availability[model]['Language'],
        quantized=quantized,
        **kwargs
    )
