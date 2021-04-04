from herpetologist import check_type

_availability = {
    'fastsep-4': {'Size (MB)': 97.8, 'SISNR PIT': 0},
    'fastsep-6': {'Size (MB)': 97.8, 'SISNR PIT': 0},
}

_availability_mel = {
    'fastsep-4': {'Size (MB)': 97.8, 'MAE PIT': 0},
    'fastsep-6': {'Size (MB)': 97.8, 'MAE PIT': 0},
}


def available_deep_wav():
    """
    List available FastSep models trained on raw 8k wav.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(_availability, text = 'Tested on 1k samples')


def available_deep_mel():
    """
    List available FastSep models trained on Melspectrogram 22k wav.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(
        _availability_mel, text = 'Tested on 1k samples'
    )


@check_type
def deep_wav(model: str = 'fastsep-4', quantized: bool = False, **kwargs):
    """
    Load FastSep model, trained on raw 8k wav using SISNR PIT loss.

    Parameters
    ----------
    model : str, optional (default='fastsep-4')
        Model architecture supported. Allowed values:

        * ``'fastsep-4'`` - FastSep 4 layers trained on raw 8k wav.
        * ``'fastsep-6'`` - FastSep 6 layers trained on raw 8k wav.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model. 
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.model.tf.Split_WAV class
    """

    model = model.lower()
    if model not in _sampling_availability:
        raise Exception(
            'model not supported, please check supported models from `malaya_speech.split_speaker.available_deep_wav()`.'
        )


@check_type
def deep_mel(model: str = 'fastsep-4', quantized: bool = False, **kwargs):
    """
    Load FastSep model, trained on Melspectrogram 22k wav using MAE PIT loss.

    Parameters
    ----------
    model : str, optional (default='fastsep-4')
        Model architecture supported. Allowed values:

        * ``'fastsep-4'`` - FastSep 4 layers trained on Melspectrogram 22k wav.
        * ``'fastsep-6'`` - FastSep 6 layers trained on Melspectrogram 22k wav.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model. 
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.model.tf.Split_MEL class
    """

    model = model.lower()
    if model not in _sampling_availability:
        raise Exception(
            'model not supported, please check supported models from `malaya_speech.split_speaker.available_deep_mel()`.'
        )
