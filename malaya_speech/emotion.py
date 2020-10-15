from malaya_speech.supervised import classification
from herpetologist import check_type

_availability = {
    'vggvox-v1': {'Size (MB)': 70.8, 'Accuracy': 0.95},
    'vggvox-v2': {'Size (MB)': 31.1, 'Accuracy': 0.9594},
    'deep-speaker': {'Size (MB)': 30.9, 'Accuracy': 0.90204},
}

labels = [
    'angry',
    'disgust',
    'fear',
    'happy',
    'sad',
    'surprise',
    'neutral',
    'not an emotion',
]


def available_model():
    """
    List available emotion detection deep models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(
        _availability, text = 'last accuracy during training session.'
    )


@check_type
def deep_model(model: str = 'vggvox-v2', **kwargs):
    """
    Load emotion detection deep model.

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
            'model not supported, please check supported models from malaya_speech.emotion.available_model()'
        )
