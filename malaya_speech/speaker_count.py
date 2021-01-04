from herpetologist import check_type

_availability = {
    'vggvox-v2': {'Size (MB)': 31.1, 'Accuracy': 0.99},
    'speakernet': {'Size (MB)': 30.9, 'Accuracy': 0.99},
}

labels = [
    '0 speaker',
    '1 speaker',
    '2 speakers',
    '3 speakers',
    '4 speakers',
    '5 speakers',
    '6 speakers',
    '7 speakers',
    '8 speakers',
    '9 speakers',
    '10 speakers',
    '11 speakers',
    '12 speakers',
    '13 speakers',
    '14 speakers',
    '15 speakers',
    'more than 15 speakers',
]


def available_model():
    """
    List available speaker counts deep models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(
        _availability, text = 'last accuracy during training session.'
    )


@check_type
def deep_model(model: str = 'vggvox-v2', quantized: bool = False, **kwargs):
    """
    Load speaker count deep model.

    Parameters
    ----------
    model : str, optional (default='vggvox-v2')
        Model architecture supported. Allowed values:

        * ``'vggvox-v2'`` - finetuned VGGVox V2.
        * ``'deep-speaker'`` - finetuned Deep Speaker.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model. 
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.supervised.classification.load function
    """

    model = model.lower()
    if model not in _availability:
        raise Exception(
            'model not supported, please check supported models from `malaya_speech.speaker_count.available_model()`.'
        )
