from herpetologist import check_type
from malaya_speech.supervised import speechsplit_conversion
from malaya_speech.utils import describe_availability

_availability = {
    'pysptk': {
        'fastspeechsplit-vggvox-v2': {
            'Size (MB)': 232,
            'Quantized Size (MB)': 59.2,
        },
        'fastspeechsplit-v2-vggvox-v2': {
            'Size (MB)': 411,
            'Quantized Size (MB)': 105,
        },
    },
    'pyworld': {
        'fastspeechsplit-vggvox-v2': {
            'Size (MB)': 232,
            'Quantized Size (MB)': 59.2,
        },
        'fastspeechsplit-v2-vggvox-v2': {
            'Size (MB)': 411,
            'Quantized Size (MB)': 105,
        },
    },
}


f0_modes = ['pysptk', 'pyworld']


def check_f0_mode(f0_mode='pysptk'):
    f0_mode = f0_mode.lower()
    if f0_mode not in f0_modes:
        raise ValueError("`f0_mode` only support one of ['pysptk', 'pyworld']")
    return f0_mode


def available_deep_conversion(f0_mode: str = 'pysptk'):
    """
    List available Voice Conversion models.

    Parameters
    ----------
    f0_mode : str, optional (default='pysptk')
        F0 conversion supported. Allowed values:

        * ``'pysptk'`` - https://github.com/r9y9/pysptk, sensitive towards gender.
        * ``'pyworld'`` - https://pypi.org/project/pyworld/
    """

    f0_mode = check_f0_mode(f0_mode=f0_mode)
    return describe_availability(_availability[f0_mode])


def deep_conversion(
    model: str = 'fastspeechsplit-v2-vggvox-v2',
    f0_mode: str = 'pysptk',
    quantized: bool = False,
    **kwargs,
):
    """
    Load Voice Conversion model.

    Parameters
    ----------
    model : str, optional (default='fastspeechsplit-v2-vggvox-v2')
        Check available models at `malaya_speech.speechsplit_conversion.available_deep_conversion(f0_mode = '{f0_mode}')`
    f0_mode : str, optional (default='pysptk')
        F0 conversion supported. Allowed values:

        * ``'pysptk'`` - https://github.com/r9y9/pysptk, sensitive towards gender.
        * ``'pyworld'`` - https://pypi.org/project/pyworld/

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.model.splitter.FastSpeechSplit class
    """

    model = model.lower()
    f0_mode = check_f0_mode(f0_mode=f0_mode)
    if model not in _availability[f0_mode]:
        raise ValueError(
            "model not supported, please check supported models from `malaya_speech.speechsplit_conversion.available_deep_conversion(f0_mode = '{f0_mode}')`."
        )

    return speechsplit_conversion.load(
        model=model,
        module=f'speechsplit-conversion-{f0_mode}',
        f0_mode=f0_mode,
        quantized=quantized,
        **kwargs,
    )
