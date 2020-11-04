from malaya_speech.path import (
    PATH_SPEECH_ENHANCEMENT,
    S3_PATH_SPEECH_ENHANCEMENT,
)
from malaya_speech.supervised import unet
from herpetologist import check_type

# https://github.com/sigsep/sigsep-mus-eval/blob/master/museval/__init__.py#L364
# Only calculate SDR, ISR, SAR on voice sample

_availability = {
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
        'Size (MB)': 96.4,
        'Quantized Size (MB)': 24.6,
        'SUM MAE': 0.8153995,
        'MAE_SPEAKER': 0.447958,
        'MAE_NOISE': 0.367441,
        'SDR': 12.349259,
        'ISR': 14.85418,
        'SAR': 15.21751,
    },
}


def available_model():
    """
    List available Speech Enhancement deep learning models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(
        _availability,
        text = 'Only calculate SDR, ISR, SAR on voice sample. Higher is better.',
    )


@check_type
def deep_model(model: str = 'resnet-unet', quantized = False, **kwargs):
    """
    Load Speech Enhancement deep learning model.

    Parameters
    ----------
    model : str, optional (default='wavenet')
        Model architecture supported. Allowed values:

        * ``'unet'`` - pretrained UNET.
        * ``'resnet-unet'`` - pretrained resnet-UNET.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model. 
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.model.tf.UNET_STFT class
    """

    model = model.lower()
    if model not in _availability:
        raise Exception(
            'model not supported, please check supported models from `malaya_speech.speech_enhancement.available_model()`.'
        )

    return unet.load_stft(
        path = PATH_SPEECH_ENHANCEMENT,
        s3_path = S3_PATH_SPEECH_ENHANCEMENT,
        model = model,
        name = 'speech-enhancement',
        instruments = ['voice', 'noise'],
        quantized = quantized,
        **kwargs
    )
