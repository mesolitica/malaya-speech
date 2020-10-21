from malaya_speech.path import (
    PATH_LANGUAGE_DETECTION,
    S3_PATH_LANGUAGE_DETECTION,
)
from malaya_speech.supervised import classification
from herpetologist import check_type

_availability = {
    'vggvox-v1': {'Size (MB)': 70.8, 'Accuracy': 0.86015},
    'vggvox-v2': {'Size (MB)': 30.9, 'Accuracy': 0.90204},
    'deep-speaker': {'Size (MB)': 96.9, 'Accuracy': 0.8945},
}

labels = [
    'english',
    'indonesian',
    'malay',
    'mandarin',
    'manglish',
    'others',
    'not a language',
]


def available_model():
    """
    List available language detection deep models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(
        _availability,
        text = 'last accuracy during training session before early stopping.',
    )


@check_type
def deep_model(model: str = 'vggvox-v2', **kwargs):
    """
    Load language detection deep model.

    Parameters
    ----------
    model : str, optional (default='vggvox-v2')
        Model architecture supported. Allowed values:

        * ``'vggvox-v1'`` - finetuned VGGVox V1.
        * ``'vggvox-v2'`` - finetuned VGGVox V2.
        * ``'deep-speaker'`` - finetuned Deep Speaker.

    Returns
    -------
    result : malaya_speech.model.tf.CLASSIFICATION class
    """
    model = model.lower()
    if model not in _availability:
        raise Exception(
            'model not supported, please check supported models from `malaya_speech.language_detection.available_model()`.'
        )

    settings = {
        'vggvox-v1': {},
        'vggvox-v2': {'concat': False},
        'deep-speaker': {'voice_only': False},
    }

    return classification.load(
        path = PATH_LANGUAGE_DETECTION,
        s3_path = S3_PATH_LANGUAGE_DETECTION,
        model = model,
        name = 'language-detection',
        extra = settings[model],
        label = labels,
        **kwargs
    )
