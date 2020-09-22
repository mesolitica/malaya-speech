from malaya_speech.path import PATH_SPEAKER_VECTOR, S3_PATH_SPEAKER_VECTOR
from malaya_speech.utils import check_file, load_graph, generate_session

_availability = {
    'vggvox-v1': ['70.8 MB', 'Embedding Size: 1024', 'EER: 0.1407'],
    'vggvox-v2': ['31.1 MB', 'Embedding Size: 512', 'EER: 0.0445'],
    'vggbox-v2-circleloss': ['31.1 MB', 'Embedding Size: 512'],
    'inception-v4-circleloss': [],
}


def available_model():
    return _availability


def load(model = 'vggvox-v2', **kwargs):
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

    check_file(
        PATH_SPEAKER_VECTOR[model], S3_PATH_SPEAKER_VECTOR[model], **kwargs
    )
    g = load_graph(PATH_SPEAKER_VECTOR[model]['model'], **kwargs)

    from malaya_speech.model.tf import SPEAKER2VEC
    from malaya_speech import featurization

    vectorizer_mapping = {
        'vggvox-v1': featurization.vggvox_v1,
        'vggvox-v2': featurization.vggvox_v2,
    }

    return SPEAKER2VEC(
        X = g.get_tensor_by_name('import/Placeholder:0'),
        logits = g.get_tensor_by_name('import/logits:0'),
        vectorizer = vectorizer_mapping[model],
        sess = generate_session(graph = g, **kwargs),
        model = model,
    )
