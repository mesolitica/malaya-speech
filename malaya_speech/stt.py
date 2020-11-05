from herpetologist import check_type

_transducer_availability = {
    'small-conformer': {'Size (MB)': 97.8, 'WER': 0, 'CER': 0},
    'base-conformer': {'Size (MB)': 97.8, 'WER': 0, 'CER': 0},
}

_ctc_availability = {
    'quartznet': {'Size (MB)': 97.8, 'WER': 0, 'CER': 0},
    'mini-jasper': {'Size (MB)': 97.8, 'WER': 0, 'CER': 0},
    'jasper': {'Size (MB)': 97.8, 'WER': 0, 'CER': 0},
}


def available_transducer():
    """
    List available Encoder-Transducer ASR models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(_transducer_availability)


def available_ctc():
    """
    List available Encoder-CTC ASR models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(_ctc_availability)


def deep_transducer(
    model: str = 'base-conformer', quantized: bool = False, **kwargs
):
    """
    Load Encoder-Transducer ASR model.

    Parameters
    ----------
    model : str, optional (default='jasper')
        Model architecture supported. Allowed values:

        * ``'small-conformer'`` - SMALL size Google Conformer, https://arxiv.org/pdf/2005.08100.pdf
        * ``'base-conformer'`` - BASE size Google Conformer, https://arxiv.org/pdf/2005.08100.pdf
        
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model. 
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.supervised.classification.load function
    """
    model = model.lower()
    if model not in _transducer_availability:
        raise Exception(
            'model not supported, please check supported models from `malaya_speech.sst.available_transducer()`.'
        )


def deep_ctc(model: str = 'jasper', quantized: bool = False, **kwargs):
    """
    Load Encoder-CTC ASR model.

    Parameters
    ----------
    model : str, optional (default='jasper')
        Model architecture supported. Allowed values:

        * ``'quartznet'`` - NVIDIA QuartzNet, https://arxiv.org/abs/1910.10261
        * ``'mini-jasper'`` - Small-factor NVIDIA Jasper, https://arxiv.org/pdf/1904.03288.pdf
        * ``'jasper'`` - NVIDIA Jasper, https://arxiv.org/pdf/1904.03288.pdf
        
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model. 
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.supervised.classification.load function
    """

    model = model.lower()
    if model not in _ctc_availability:
        raise Exception(
            'model not supported, please check supported models from `malaya_speech.sst.available_ctc()`.'
        )
