from malaya_speech.path import (
    PATH_SPEECH_ENHANCEMENT,
    S3_PATH_SPEECH_ENHANCEMENT,
)
from malaya_speech.supervised import unet
from herpetologist import check_type

# https://github.com/sigsep/sigsep-mus-eval/blob/master/museval/__init__.py#L364
# Only calculate SDR, ISR, SAR on voice sample

_masking_availability = {
    'unet': {
        'Size (MB)': 78.9,
        'Quantized Size (MB)': 20,
        'SUM MAE': 0.85896,
        'MAE_SPEAKER': 0.46849,
        'MAE_NOISE': 0.39046,
        'SDR': 12.12805,
        'ISR': 14.67067,
        'SAR': 15.019682,
    },
    'resnet-unet': {
        'Size (MB)': 91.4,
        'Quantized Size (MB)': 23,
        'SUM MAE': 0.8153995,
        'MAE_SPEAKER': 0.447958,
        'MAE_NOISE': 0.367441,
        'SDR': 12.349259,
        'ISR': 14.85418,
        'SAR': 15.21751,
    },
}

_sampling_availability = {
    'unet-enhance-24': {
        'Size (MB)': 40.7,
        'Quantized Size (MB)': 10.3,
        'SDR': 9.877178,
        'ISR': 15.916217,
        'SAR': 13.709130,
    },
    'unet-enhance-36': {
        'Size (MB)': 91.3,
        'Quantized Size (MB)': 23.4,
        'SDR': 8.749694,
        'ISR': 14.512658,
        'SAR': 13.963656,
    },
}


def available_deep_masking():
    """
    List available Speech Enhancement STFT masking deep learning model.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(
        _masking_availability,
        text = 'Only calculate SDR, ISR, SAR on voice sample. Higher is better.',
    )


def available_deep_enhance():
    """
    List available Speech Enhancement UNET Waveform sampling deep learning model.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(
        _sampling_availability,
        text = 'Only calculate SDR, ISR, SAR on voice sample. Higher is better.',
    )


@check_type
def deep_masking(model: str = 'resnet-unet', quantized: bool = False, **kwargs):
    """
    Load Speech Enhancement STFT UNET masking deep learning model.

    Parameters
    ----------
    model : str, optional (default='resnet-unet')
        Model architecture supported. Allowed values:

        * ``'unet'`` - pretrained UNET.
        * ``'resnet-unet'`` - pretrained resnet-UNET.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model. 
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.model.tf.UNETSTFT class
    """

    model = model.lower()
    if model not in _masking_availability:
        raise Exception(
            'model not supported, please check supported models from `malaya_speech.speech_enhancement.available_deep_masking()`.'
        )

    return unet.load_stft(
        path = PATH_SPEECH_ENHANCEMENT['masking'],
        s3_path = S3_PATH_SPEECH_ENHANCEMENT['masking'],
        model = model,
        name = 'speech-enhancement',
        instruments = ['voice', 'noise'],
        quantized = quantized,
        **kwargs
    )


@check_type
def deep_enhance(
    model: str = 'unet-enhance-24', quantized: bool = False, **kwargs
):
    """
    Load Speech Enhancement UNET Waveform sampling deep learning model.

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
            'model not supported, please check supported models from `malaya_speech.speech_enhancement.available_deep_enhance()`.'
        )

    return unet.load_1d(
        path = PATH_SPEECH_ENHANCEMENT['enhance'],
        s3_path = S3_PATH_SPEECH_ENHANCEMENT['enhance'],
        model = model,
        name = 'speech-enhancement',
        quantized = quantized,
        **kwargs
    )
