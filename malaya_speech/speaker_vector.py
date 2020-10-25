from malaya_speech.path import PATH_SPEAKER_VECTOR, S3_PATH_SPEAKER_VECTOR
from malaya_speech.supervised import classification
from herpetologist import check_type

_availability = {
    'deep-speaker': {'Size (MB)': 96.7, 'Embedding Size': 512, 'EER': 0.2187},
    'vggvox-v1': {'Size (MB)': 70.8, 'Embedding Size': 1024, 'EER': 0.1407},
    'vggvox-v2': {'Size (MB)': 43.2, 'Embedding Size': 512, 'EER': 0.0445},
    'speakernet': {'Size (MB)': 35, 'Embedding Size': 7205, 'EER': 0.02122},
}


def available_model():
    """
    List available speaker vector deep models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(
        _availability,
        text = 'tested on VoxCeleb2 test set. Lower EER is better.',
    )


@check_type
def deep_model(model: str = 'speakernet', **kwargs):
    """
    Load Speaker2Vec model.

    Parameters
    ----------
    model : str, optional (default='speakernet')
        Model architecture supported. Allowed values:

        * ``'vggvox-v1'`` - VGGVox V1, embedding size 1024, exported from https://github.com/linhdvu14/vggvox-speaker-identification
        * ``'vggvox-v2'`` - VGGVox V2, embedding size 512, exported from https://github.com/WeidiXie/VGG-Speaker-Recognition
        * ``'deep-speaker'`` - Deep Speaker, embedding size 512, exported from https://github.com/philipperemy/deep-speaker
        * ``'speakernet'`` - SpeakerNet, embedding size 7205, exported from https://github.com/NVIDIA/NeMo/tree/main/examples/speaker_recognition

    Returns
    -------
    result : malaya_speech.model.tf.SPEAKER2VEC class
    """

    model = model.lower()
    if model not in _availability:
        raise Exception(
            'model not supported, please check supported models from `malaya_speech.speaker_vector.available_model()`.'
        )

    return classification.load(
        path = PATH_SPEAKER_VECTOR,
        s3_path = S3_PATH_SPEAKER_VECTOR,
        model = model,
        name = 'speaker-vector',
        extra = {},
        label = {},
        **kwargs
    )
