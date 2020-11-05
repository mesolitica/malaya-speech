from malaya_speech.path import PATH_GENDER, S3_PATH_GENDER
from malaya_speech.supervised import classification
from herpetologist import check_type

_availability = {
    'vggvox-v2': {
        'Size (MB)': 31.1,
        'Quantized Size (MB)': 7.92,
        'Accuracy': 0.9756,
    },
    'deep-speaker': {
        'Size (MB)': 96.9,
        'Quantized Size (MB)': 24.4,
        'Accuracy': 0.9455,
    },
}

labels = ['male', 'female', 'not a gender']


def available_model():
    """
    List available gender detection deep models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(
        _availability,
        text = 'last accuracy during training session before early stopping.',
    )


@check_type
def deep_model(model: str = 'vggvox-v2', quantized: bool = False, **kwargs):
    """
    Load gender detection deep model.

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
            'model not supported, please check supported models from `malaya_speech.gender.available_model()`.'
        )

    settings = {
        'vggvox-v2': {'concat': False},
        'deep-speaker': {'voice_only': False},
    }

    return classification.load(
        path = PATH_GENDER,
        s3_path = S3_PATH_GENDER,
        model = model,
        name = 'gender',
        extra = settings[model],
        label = labels,
        quantized = quantized,
        **kwargs
    )
