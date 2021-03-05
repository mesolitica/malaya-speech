from herpetologist import check_type

_availability = {
    'vggvox-v2': {
        'Size (MB)': 31.1,
        'Quantized Size (MB)': 7.92,
        'Accuracy': 0.89912,
    },
    'speakernet': {
        'Size (MB)': 20.3,
        'Quantized Size (MB)': 5.18,
        'Accuracy': 0.79875,
    },
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
    'more than 10 speakers',
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
        * ``'speakernet'`` - finetuned SpeakerNet.
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

    settings = {
        'vggvox-v2': {'hop_length': 300, 'concat': False, 'mode': 'eval'},
        'speakernet': {'frame_ms': 20, 'stride_ms': 12.0},
    }

    return classification.load(
        model = model,
        module = 'speaker-count',
        extra = settings[model],
        label = labels,
        quantized = quantized,
        **kwargs
    )
