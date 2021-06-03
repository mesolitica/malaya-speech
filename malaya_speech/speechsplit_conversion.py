from herpetologist import check_type
from malaya_speech.supervised import speechsplit_conversion

_availability = {
    'pysptk': {
        'fastspeechsplit-vggvox-v2': {
            'Size (MB)': 232,
            'Quantized Size (MB)': 59.2,
        },
        'fastspeechsplit-v2-vggvox-v2': {
            'Size (MB)': 105,
            'Quantized Size (MB)': 411,
        },
    },
    'pyworld': {
        'fastspeechsplit-vggvox-v2': {
            'Size (MB)': 232,
            'Quantized Size (MB)': 59.2,
        },
        'fastspeechsplit-v2-vggvox-v2': {
            'Size (MB)': 105,
            'Quantized Size (MB)': 411,
        },
    },
}


f0_modes = ['pysptk', 'pyworld']


def check_f0_mode(f0_mode='pysptk'):
    f0_mode = f0_mode.lower()
    if f0_mode not in f0_modes:
        raise ValueError("`f0_mode` only support one of ['pysptk', 'pyworld']")
    return f0_mode


def available_deep_conversion(f0_mode='pysptk'):
    """
    List available Voice Conversion models.
    """
    from malaya_speech.utils import describe_availability

    f0_mode = check_f0_mode(f0_mode=f0_mode)
    return describe_availability(_availability[f0_mode])


def deep_conversion(
    model: str = 'fastspeechsplit-v2-vggvox-v2',
    f0_mode='pysptk',
    quantized: bool = False,
    **kwargs,
):
    """
    Load Voice Conversion model.

    Parameters
    ----------
    model : str, optional (default='fastvc-32-vggvox-v2')
        Model architecture supported. Allowed values:

        * ``'fastspeechsplit-vggvox-v2'`` - FastSpeechSplit with VGGVox-v2 Speaker Vector.
        * ``'fastspeechsplit-v2-vggvox-v2'`` - FastSpeechSplit V2 with VGGVox-v2 Speaker Vector.

    f0_mode : str, optional (default='pysptk)
        F0 conversion supported. Allowed values:

        * ``'pysptk'`` - https://github.com/r9y9/pysptk, sensitive towards gender.
        * ``'pyworld'`` - https://pypi.org/project/pyworld/

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.supervised.speechsplit_conversion.load function
    """

    model = model.lower()
    f0_mode = check_f0_mode(f0_mode=f0_mode)
    if model not in _availability[f0_mode]:
        raise ValueError(
            "model not supported, please check supported models from `malaya_speech.speechsplit_conversion.available_deep_conversion(f0_mode = '{f0_mode}')`."
        )

    if f0_mode == 'pysptk':
        try:
            from pysptk import sptk

        except BaseException:
            raise ModuleNotFoundError(
                'pysptk not installed. Please install it by `pip install pysptk` and try again.'
            )

    if f0_mode == 'pyworld':
        try:
            import pyworld as pw

        except BaseException:
            raise ModuleNotFoundError(
                'pyworld not installed. Please install it by `pip install pyworld` and try again.'
            )

    return speechsplit_conversion.load(
        model=model,
        module=f'speechsplit-conversion-{f0_mode}',
        f0_mode=f0_mode,
        quantized=quantized,
        **kwargs,
    )
