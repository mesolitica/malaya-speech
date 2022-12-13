from herpetologist import check_type
from malaya_speech.supervised import voice_conversion
from malaya_speech.utils import describe_availability

_availability = {
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


def available_deep_conversion():
    """
    List available Voice Conversion models.
    """

    return describe_availability(_availability)


def deep_conversion(
    model: str = 'fastvc-32-vggvox-v2', quantized: bool = False, **kwargs
):
    """
    Load Voice Conversion model.

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
