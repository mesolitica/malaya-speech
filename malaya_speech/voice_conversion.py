from herpetologist import check_type
from malaya_speech.supervised import voice_conversion
from malaya_speech.utils import describe_availability
import logging
import warnings

_availability_fastvc = {
    'fastvc-32-vggvox-v2': {
        'Size (MB)': 190,
        'Quantized Size (MB)': 54.1,
        'Total loss': 0.2851,
    },
    'fastvc-64-vggvox-v2': {
        'Size (MB)': 194,
        'Quantized Size (MB)': 55.7,
        'Total loss': 0.2764,
    },
}


def available_fastvc():
    """
    List available Voice Conversion models.
    """
    warnings.warn(
        '`malaya_speech.voice_conversion.available_fastvc` is using Tensorflow, malaya-speech no longer improved it after version 1.4.0', DeprecationWarning)

    return describe_availability(_availability)


def fastvc(
    model: str = 'fastvc-32-vggvox-v2', quantized: bool = False, **kwargs
):
    """
    Load Voice Conversion FastVC model.

    Parameters
    ----------
    model : str, optional (default='fastvc-32-vggvox-v2')
        Check available models at `malaya_speech.voice_conversion.available_deep_conversion()`.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.model.synthesis.FastVC class
    """

    warnings.warn(
        '`malaya_speech.voice_conversion.fastvc` is using Tensorflow, malaya-speech no longer improved it after version 1.4.0', DeprecationWarning)

    model = model.lower()
    if model not in _availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.voice_conversion.available_deep_conversion()`.'
        )

    return voice_conversion.load(
        model=model,
        module='voice-conversion',
        quantized=quantized,
        **kwargs
    )
