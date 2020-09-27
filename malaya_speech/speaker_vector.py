from malaya_speech.path import PATH_SPEAKER_VECTOR, S3_PATH_SPEAKER_VECTOR
from malaya_speech.supervised import classification
from herpetologist import check_type

_availability = {
    'vggvox-v1': {'Size (MB)': 70.8, 'Embedding Size': 1024, 'EER': 0.1407},
    'vggvox-v2': {'Size (MB)': 31.1, 'Embedding Size': 512, 'EER': 0.0445},
}


def available_model():
    """
    List available speaker vector deep models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(_availability)


@check_type
def deep_model(model: str = 'vggvox-v2', **kwargs):
    """
    Load Speaker2Vec model.

    Parameters
    ----------
    model : str, optional (default='vggvox-v2')
        Model architecture supported. Allowed values:

        * ``'vggvox-v1'`` - VGGVox V1, embedding size 1024.
        * ``'vggvox-v2'`` - VGGVox V2, embedding size 512.

    Returns
    -------
    result : malaya_speech.model.tf.SPEAKER2VEC class
    """

    model = model.lower()
    if model not in _availability:
        raise Exception(
            'model not supported, please check supported models from malaya_speech.speaker_vector.available_model()'
        )

    return classification.load(
        PATH_SPEAKER_VECTOR,
        S3_PATH_SPEAKER_VECTOR,
        model,
        'speaker-vector',
        {},
        {},
    )
