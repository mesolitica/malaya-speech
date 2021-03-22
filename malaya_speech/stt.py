from malaya_speech.supervised import stt
from herpetologist import check_type
import json

_transducer_availability = {
    'small-conformer': {
        'Size (MB)': 49.2,
        'Quantized Size (MB)': 18.1,
        'WER': 0.23728,
        'CER': 0.08524,
    },
    'conformer': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 37.1,
        'WER': 0.22465,
        'CER': 0.07985,
    },
    'large-conformer': {
        'Size (MB)': 399,
        'Quantized Size (MB)': 103,
        'WER': 0.23261,
        'CER': 0.08241,
    },
    'alconformer': {
        'Size (MB)': 38,
        'Quantized Size (MB)': 14.9,
        'WER': 0.25893,
        'CER': 0.09803,
    },
    'large-alconformer': {
        'Size (MB)': 58.8,
        'Quantized Size (MB)': 20.2,
        'WER': 0.2377,
        'CER': 0.08756,
    },
}

google_accuracy = {
    'WER': 0.14270,
    'CER': 0.04682,
    'last update': '2021-03-11',
    'library use': 'https://pypi.org/project/SpeechRecognition/',
    'notebook link': 'https://github.com/huseinzol05/malaya-speech/blob/master/data/semisupervised-audiobook/benchmark-google-speech-malaya-speech-test-dataset.ipynb',
}


def available_transducer():
    """
    List available Encoder-Transducer ASR models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(_transducer_availability)


@check_type
def deep_transducer(
    model: str = 'conformer', quantized: bool = False, **kwargs
):
    """
    Load Encoder-Transducer ASR model.

    Parameters
    ----------
    model : str, optional (default='jasper')
        Model architecture supported. Allowed values:

        * ``'small-conformer'`` - SMALL size Google Conformer, https://arxiv.org/pdf/2005.08100.pdf
        * ``'conformer'`` - BASE size Google Conformer, https://arxiv.org/pdf/2005.08100.pdf
        * ``'large-conformer'`` - LARGE size Google Conformer, https://arxiv.org/pdf/2005.08100.pdf
        * ``'alconformer'`` - BASE size A-Lite Google Conformer.
        * ``'large-alconformer'`` - LARGE size A-Lite Google Conformer.
        
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model. 
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.model.tf.Transducer class
    """
    model = model.lower()
    if model not in _transducer_availability:
        raise Exception(
            'model not supported, please check supported models from `malaya_speech.stt.available_transducer()`.'
        )

    return stt.transducer_load(
        model = model,
        module = 'speech-to-text',
        quantized = quantized,
        **kwargs
    )
