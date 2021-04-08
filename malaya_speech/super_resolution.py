from malaya_speech.supervised import unet
from herpetologist import check_type

_availability = {
    'srgan-128': {
        'Size (MB)': 7.37,
        'Quantized Size (MB)': 2.04,
        'SDR': 17.03345,
        'ISR': 22.33026,
        'SAR': 17.454372,
    },
    'srgan-256': {
        'Size (MB)': 29.5,
        'Quantized Size (MB)': 7.55,
        'SDR': 16.34558,
        'ISR': 22.067493,
        'SAR': 17.02439,
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

        * ``'srgan-128'`` - srgan with 128 filter size and 16 residual blocks.
        * ``'srgan-256'`` - srgan with 256 filter size and 16 residual blocks.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model. 
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.model.tf.UNET1D class
    """
    model = model.lower()
    if model not in _availability:
        raise Exception(
            'model not supported, please check supported models from `malaya_speech.super_resolution.available_model()`.'
        )
    return unet.load_1d(
        model = model,
        module = 'super-resolution',
        quantized = quantized,
        **kwargs
    )
