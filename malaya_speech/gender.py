from malaya_speech.path import PATH_GENDER, S3_PATH_GENDER
from malaya_speech.supervised import classification
from herpetologist import check_type

_availability = {
    'vggvox-v2': {'Size (MB)': 31.1, 'Accuracy': 0.9756},
    'deep-speaker': {'Size (MB)': 96.9, 'Accuracy': 0.9455},
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
def deep_model(model: str = 'vggvox-v2', **kwargs):
    """
    Load gender detection deep model.

    Parameters
    ----------
    model : str, optional (default='vggvox-v2')
        Model architecture supported. Allowed values:

        * ``'vggvox-v2'`` - finetuned VGGVox V2.
        * ``'deep-speaker'`` - finetuned Deep Speaker.

    Returns
    -------
    result : malaya_speech.model.tf.CLASSIFICATION class
    """
    model = model.lower()
    if model not in _availability:
        raise Exception(
            'model not supported, please check supported models from `malaya_speech.gender.available_model()`.'
        )

    settings = {
        'vggvox-v1': {},
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
        **kwargs
    )
