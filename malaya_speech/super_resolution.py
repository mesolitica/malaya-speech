from malaya_speech.supervised import unet as load_unet, enhancement as load_enhancement
from malaya_speech.utils import describe_availability
from herpetologist import check_type
import logging

logger = logging.getLogger(__name__)

_availability_unet = {
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

_availability_tfgan = {
    'voicefixer': {
        'Size (MB)': 489.0,
    },
    'nvsr': {
        'Size (MB)': 468.0,
    }
}


def available_unet():
    """
    List available Super Resolution 4x deep learning UNET models.
    """
    logger.info('Only calculate SDR, ISR, SAR on voice sample. Higher is better.')

    return describe_availability(_availability_unet)


def available_tfgan():
    """
    List available Super Resolution deep learning UNET + TFGAN Vocoder models.
    """
    logger.info('Only calculate SDR, ISR, SAR on voice sample. Higher is better.')

    return describe_availability(_availability_tfgan)


@check_type
def unet(model: str = 'srgan-256', quantized: bool = False, **kwargs):
    """
    Load Super Resolution 4x deep learning UNET model.

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
    if model not in _availability_unet:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.super_resolution.available_unet()`.'
        )
    return load_unet.load_1d(
        model=model,
        module='super-resolution',
        quantized=quantized,
        **kwargs
    )


def tfgan(model: str = 'voicefixer', **kwargs):
    """
    Load TFGAN based Speech Resolution.

    Parameters
    ----------
    model : str, optional (default='voicefixer')
        Model architecture supported. Allowed values:

        * ``'voicefixer'`` - originally from https://github.com/haoheliu/voicefixer.
        * ``'nvsr'`` - originally from https://github.com/haoheliu/ssr_eval/tree/main/examples/NVSR.

    Returns
    -------
    result : malaya_speech.torch_model.super_resolution.VoiceFixer
    """
    model = model.lower()
    if model not in _availability_tfgan:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.super_resolution.available_tfgan()`.'
        )

    return load_enhancement.load_tfgan(model=model)
