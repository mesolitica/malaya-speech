from malaya_speech.supervised import stt
from herpetologist import check_type
import json

_transducer_availability = {
    'small-conformer': {
        'Size (MB)': 44.1,
        'Quantized Size (MB)': 13.3,
        'WER': 0.2692,
        'CER': 0.1058,
    },
    'conformer': {
        'Size (MB)': 120,
        'Quantized Size (MB)': 32.7,
        'WER': 0.2442,
        'CER': 0.0910,
    },
    'large-conformer': {
        'Size (MB)': 399,
        'Quantized Size (MB)': 103,
        'WER': 0.2390,
        'CER': 0.0812,
    },
    'alconformer': {
        'Size (MB)': 33.2,
        'Quantized Size (MB)': 10.5,
        'WER': 0.30567,
        'CER': 0.12267,
    },
    'large-alconformer': {
        'Size (MB)': 33.2,
        'Quantized Size (MB)': 10.5,
        'WER': 0.30567,
        'CER': 0.12267,
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
    model : str, optional (default='jasper')
        Model architecture supported. Allowed values:

        * ``'small-conformer'`` - SMALL size Google Conformer, https://arxiv.org/pdf/2005.08100.pdf
        * ``'conformer'`` - BASE size Google Conformer, https://arxiv.org/pdf/2005.08100.pdf
        * ``'large-conformer'`` - LARGE size Google Conformer, https://arxiv.org/pdf/2005.08100.pdf
        * ``'alconformer'`` - BASE size A-Lite Google Conformer.
        
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
