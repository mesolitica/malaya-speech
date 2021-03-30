from malaya_speech.supervised import stt
from herpetologist import check_type
import json

_transducer_availability = {
    'small-conformer': {
        'Size (MB)': 49.2,
        'Quantized Size (MB)': 18.1,
        'WER': 0.23582,
        'CER': 0.08771,
    },
    'conformer': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 37.1,
        'WER': 0.21718,
        'CER': 0.07562,
    },
    'large-conformer': {
        'Size (MB)': 399,
        'Quantized Size (MB)': 103,
        'WER': 0.21938,
        'CER': 0.07306,
    },
    'small-alconformer': {
        'Size (MB)': 38,
        'Quantized Size (MB)': 14.9,
        'WER': 0.25893,
        'CER': 0.09803,
    },
    'alconformer': {
        'Size (MB)': 38,
        'Quantized Size (MB)': 14.9,
        'WER': 0.25893,
        'CER': 0.09803,
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
    model : str, optional (default='conformer')
        Model architecture supported. Allowed values:

        * ``'small-conformer'`` - SMALL size Google Conformer with Pretrained LM Malay language.
        * ``'conformer'`` - BASE size Google Conformer with Pretrained LM Malay language.
        * ``'large-conformer'`` - LARGE size Google Conformer with Pretrained LM Malay language.
        * ``'small-alconformer'`` - SMALL size A-Lite Google Conformer with Pretrained LM Malay language.
        * ``'alconformer'`` - BASE size A-Lite Google Conformer with Pretrained LM Malay language.
        
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
