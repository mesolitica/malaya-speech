from malaya_speech.supervised import unet
from herpetologist import check_type

# https://github.com/sigsep/sigsep-mus-eval/blob/master/museval/__init__.py#L364
# Only calculate SDR, ISR, SAR on voice sample

_sampling_availability = {
    'unet': {
        'Size (MB)': 40.7,
        'Quantized Size (MB)': 10.3,
        'SDR': 9.877178,
        'ISR': 15.916217,
        'SAR': 13.709130,
    },
    'resnext-unet': {
        'Size (MB)': 91.3,
        'Quantized Size (MB)': 23.4,
        'SDR': 8.749694,
        'ISR': 14.512658,
        'SAR': 13.963656,
    },
}


def available_deep_enhance():
    """
    List available Background Enhance UNET Waveform sampling deep learning model.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(
        _sampling_availability,
        text = 'Only calculate SDR, ISR, SAR on voice sample. Higher is better.',
    )


@check_type
def deep_enhance(
    model: str = 'unet-enhance-24', quantized: bool = False, **kwargs
):
    """
    Load Background Enhance UNET Waveform sampling deep learning model.

    Parameters
    ----------
    model : str, optional (default='unet-enhance-24')
        Model architecture supported. Allowed values:

        * ``'unet-enhance-24'`` - pretrained UNET Speech Enhancement 24 filter size.
        * ``'unet-enhance-36'`` - pretrained UNET Speech Enhancement 36 filter size.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model. 
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.model.tf.UNET1D class
    """

    model = model.lower()
    if model not in _sampling_availability:
        raise Exception(
            'model not supported, please check supported models from `malaya_speech.background_enhance.available_deep_enhance()`.'
        )

    return unet.load_1d(
        model = model,
        module = 'background-enhance',
        quantized = quantized,
        **kwargs
    )
