from malaya_speech.path import PATH_SPEAKER_VECTOR, S3_PATH_SPEAKER_VECTOR
from malaya_speech.utils import check_file, load_graph, generate_session

_availability = {
    'vggvox-v1': ['70.8 MB'],
    'vggvox-v2': ['31.1 MB'],
    'vggbox-v2-circleloss': ['31.1 MB'],
}


def available_model():
    return _availability


def load(model = 'pretrained-vggvox-v2', **kwargs):
    """
    Load Speaker2Vec model.

    Parameters
    ----------
    model : str, optional (default='pretrained-vggvox-v2')
        Model architecture supported. Allowed values:

        * ``'pretrained-vggvox-v1'`` - VGGVox V1.
        * ``'pretrained-vggvox-v2'`` - VGGVox V2.

    Returns
    -------
    result : malaya.model.tf.CONSTITUENCY class
    """

    model = model.lower()
    if model not in _availability:
        raise Exception(
            'model not supported, please check supported models from malaya_speech.speaker_vector.available_model()'
        )

    check_file(
        PATH_SPEAKER_VECTOR[model], S3_PATH_SPEAKER_VECTOR[model], **kwargs
    )
    g = load_graph(PATH_SPEAKER_VECTOR[model]['model'], **kwargs)
