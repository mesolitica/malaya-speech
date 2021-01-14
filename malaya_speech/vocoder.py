from malaya_speech.path import (
    PATH_VOCODER_MELGAN,
    PATH_VOCODER_MBMELGAN,
    S3_PATH_VOCODER_MELGAN,
    S3_PATH_VOCODER_MBMELGAN,
)
from malaya_speech.supervised import vocoder
from herpetologist import check_type

_melgan_availability = {
    'male': {
        'Size (MB)': 17.3,
        'Quantized Size (MB)': 4.53,
        'Mel loss': 0.4443,
    },
    'female': {
        'Size (MB)': 17.3,
        'Quantized Size (MB)': 4.53,
        'Mel loss': 0.4434,
    },
    'husein': {
        'Size (MB)': 17.3,
        'Quantized Size (MB)': 4.53,
        'Mel loss': 0.4442,
    },
    'haqkiem': {
        'Size (MB)': 17.3,
        'Quantized Size (MB)': 4.53,
        'Mel loss': 0.4819,
    },
}
_mbmelgan_availability = {
    'female': {
        'Size (MB)': 10.4,
        'Quantized Size (MB)': 2.82,
        'Mel loss': 0.4356,
    },
    'male': {
        'Size (MB)': 10.4,
        'Quantized Size (MB)': 2.82,
        'Mel loss': 0.3735,
    },
    'husein': {
        'Size (MB)': 10.4,
        'Quantized Size (MB)': 2.82,
        'Mel loss': 0.4356,
    },
    'haqkiem': {
        'Size (MB)': 10.4,
        'Quantized Size (MB)': 2.82,
        'Mel loss': 0.4192,
    },
}


def available_melgan():
    """
    List available MelGAN Mel-to-Speech models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(_melgan_availability)


def available_mbmelgan():
    """
    List available Multiband MelGAN Mel-to-Speech models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(_mbmelgan_availability)


@check_type
def melgan(model: str = 'female', quantized: bool = False, **kwargs):
    """
    Load MelGAN Vocoder model.

    Parameters
    ----------
    model : str, optional (default='jasper')
        Model architecture supported. Allowed values:

        * ``'female'`` - MelGAN trained on female voice.
        * ``'male'`` - MelGAN trained on male voice.
        * ``'husein'`` - MelGAN trained on Husein voice, https://www.linkedin.com/in/husein-zolkepli/
        * ``'haqkiem'`` - MelGAN trained on Haqkiem voice, https://www.linkedin.com/in/haqkiem-daim/
        
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model. 
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.supervised.vocoder.load function
    """
    model = model.lower()
    if model not in _melgan_availability:
        raise Exception(
            'model not supported, please check supported models from `malaya_speech.vocoder.available_melgan()`.'
        )

    return vocoder.load(
        path = PATH_VOCODER_MELGAN,
        s3_path = S3_PATH_VOCODER_MELGAN,
        model = model,
        name = 'vocoder',
        quantized = quantized,
        **kwargs
    )


@check_type
def mbmelgan(model: str = 'female', quantized: bool = False, **kwargs):
    """
    Load Multiband MelGAN Vocoder model.

    Parameters
    ----------
    model : str, optional (default='jasper')
        Model architecture supported. Allowed values:

        * ``'female'`` - MelGAN trained on female voice.
        * ``'male'`` - MelGAN trained on male voice.
        * ``'husein'`` - MelGAN trained on Husein voice, https://www.linkedin.com/in/husein-zolkepli/
        * ``'haqkiem'`` - MelGAN trained on Haqkiem voice, https://www.linkedin.com/in/haqkiem-daim/
        
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model. 
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.supervised.vocoder.load function
    """
    model = model.lower()
    if model not in _mbmelgan_availability:
        raise Exception(
            'model not supported, please check supported models from `malaya_speech.vocoder.available_mbmelgan()`.'
        )
    return vocoder.load(
        path = PATH_VOCODER_MBMELGAN,
        s3_path = S3_PATH_VOCODER_MBMELGAN,
        model = model,
        name = 'vocoder',
        quantized = quantized,
        **kwargs
    )
