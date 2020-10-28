from malaya_speech.path import PATH_SPEAKER_CHANGE, S3_PATH_SPEAKER_CHANGE
from malaya_speech.supervised import classification
from herpetologist import check_type

_availability = {
    'vggvox-v2': {'Size (MB)': 31.1, 'Accuracy': 0.9594},
    'speakernet': {'Size (MB)': 30.9, 'Accuracy': 0.99},
}


def available_model():
    """
    List available speaker change deep models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(_availability)


@check_type
def deep_model(model: str = 'vggvox-v2', **kwargs):
    """
    Load speaker change deep model.

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
            'model not supported, please check supported models from `malaya_speech.speaker_change.available_model()`.'
        )

    settings = {
        'vggvox-v2': {'hop_length': 24, 'concat': False, 'mode': 'eval'},
        'speakernet': {'frame_ms': 20, 'stride_ms': 0.3},
    }

    return classification.load(
        path = PATH_SPEAKER_CHANGE,
        s3_path = S3_PATH_SPEAKER_CHANGE,
        model = model,
        name = 'speaker-change',
        extra = settings[model],
        label = [False, True],
        **kwargs
    )
