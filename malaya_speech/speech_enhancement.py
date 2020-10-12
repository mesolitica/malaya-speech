from malaya_speech.path import (
    PATH_SPEECH_ENHANCEMENT,
    S3_PATH_SPEECH_ENHANCEMENT,
)
from malaya_speech.supervised import unet
from herpetologist import check_type

_unet_availability = {
    'resnet34': {'Size (MB)': 97.8, 'MSE': 0.0003},
    'inception-v3': {'Size (MB)': 120, 'MSE': 0.0003},
}

_vocoder_availability = {'wavenet': {'Size (MB)': 70.8, 'MSE': 0.0003}}


def available_unet():
    """
    List available Speech Enhancement UNET models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(_unet_availability)


def available_vocoder():
    """
    List available Speech Enhancement UNET models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(_vocoder_availability)


@check_type
def unet(model: str = 'resnet34', **kwargs):
    """
    Load Speech Enhancement UNET model.

    Parameters
    ----------
    model : str, optional (default='wavenet')
        Model architecture supported. Allowed values:

        * ``'resnet34'`` - pretrained resnet34 UNET.
        * ``'inception-v3'`` - pretrained inception V3 UNET.

    Returns
    -------
    result : malaya_speech.model.tf.UNET class
    """

    model = model.lower()
    if model not in _unet_availability:
        raise Exception(
            'model not supported, please check supported models from malaya_speech.speech_enhancement.available_unet()'
        )

    return unet.load(
        path = PATH_SPEECH_ENHANCEMENT,
        s3_path = S3_PATH_SPEECH_ENHANCEMENT,
        model = model,
        name = 'speech-enhancement',
        **kwargs
    )


@check_type
def vocoder(model: str = 'wavenet', **kwargs):
    """
    Load Speech Enhancement vocoder model.

    Parameters
    ----------
    model : str, optional (default='wavenet')
        Model architecture supported. Allowed values:

        * ``'wavenet'`` - pretrained wavenet.

    Returns
    -------
    result : malaya_speech.model.tf.VOCODER class
    """

    model = model.lower()
    if model not in _vocoder_availability:
        raise Exception(
            'model not supported, please check supported models from malaya_speech.speech_enhancement.available_vocoder()'
        )
