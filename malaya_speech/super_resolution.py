from herpetologist import check_type

_availability = {
    'sr-256': {
        'Size (MB)': 78.9,
        'Quantized Size (MB)': 20,
        'SDR': 12.12805,
        'ISR': 14.67067,
        'SAR': 15.019682,
    },
    'sr-512': {
        'Size (MB)': 78.9,
        'Quantized Size (MB)': 20,
        'SDR': 12.12805,
        'ISR': 14.67067,
        'SAR': 15.019682,
    },
}


def available_model():
    """
    List available Super Resolution 4x deep learning models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(
        _availability,
        text = 'Only calculate SDR, ISR, SAR on voice sample. Higher is better.',
    )


@check_type
def deep_model(model: str = 'srgan-256', quantized: bool = False, **kwargs):
    """
    Load Super Resolution 4x deep learning model.

    Parameters
    ----------
    model : str, optional (default='srgan-256')
        Model architecture supported. Allowed values:

        * ``'srgan-256'`` - srgan with 256 filter size and 16 residual blocks.
        * ``'srgan-512'`` - srgan with 512 filter size and 16 residual blocks.
        * ``'srgan-768'`` - srgan with 768 filter size and 16 residual blocks.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model. 
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.model.tf.SUPER_RESOLUTION class
    """
    model = model.lower()
    if model not in _availability:
        raise Exception(
            'model not supported, please check supported models from `malaya_speech.super_resolution.available_model()`.'
        )
