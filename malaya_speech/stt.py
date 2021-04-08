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
        'Size (MB)': 404,
        'Quantized Size (MB)': 107,
        'WER': 0.21938,
        'CER': 0.07306,
    },
    'small-alconformer': {
        'Size (MB)': 18.8,
        'Quantized Size (MB)': 10.1,
        'WER': 0.30373,
        'CER': 0.12471,
    },
    'alconformer': {
        'Size (MB)': 38,
        'Quantized Size (MB)': 14.9,
        'WER': 0.25611,
        'CER': 0.09726,
    },
    'small-conformer-mixed': {
        'Size (MB)': 49.2,
        'Quantized Size (MB)': 18.1,
        'WER': 0.43149,
        'CER': 0.29467,
    },
    'conformer-mixed': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 37.1,
        'WER': 0.35191,
        'CER': 0.23667,
    },
    'large-conformer-mixed': {
        'Size (MB)': 404,
        'Quantized Size (MB)': 107,
        'WER': 0.35191,
        'CER': 0.23667,
    },
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
        * ``'small-conformer-mixed'`` - SMALL size Google Conformer with Pretrained LM Mixed (Malay + Singlish) languages.
        * ``'conformer-mixed'`` - BASE size Google Conformer with Pretrained LM Mixed (Malay + Singlish) languages.
        * ``'large-conformer-mixed'`` - LARGE size Google Conformer with Pretrained LM Mixed (Malay + Singlish) languages.
        * ``'small-conformer-singlish'`` - SMALL size Google Conformer with Pretrained LM Singlish language.
        * ``'conformer-singlish'`` - BASE size Google Conformer with Pretrained LM Singlish language.
        * ``'large-conformer-singlish'`` - LARGE size Google Conformer with Pretrained LM Singlish language.
        
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
