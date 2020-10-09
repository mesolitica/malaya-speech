from herpetologist import check_type

_availability = {
    'vggvox-v1': {'Size (MB)': 70.8, 'Embedding Size': 1024, 'EER': 0.1407},
    'vggvox-v2': {'Size (MB)': 43.2, 'Embedding Size': 512, 'EER': 0.0445},
    'inception-v4': {'Size (MB)': 181, 'Embedding Size': 512, 'EER': 0.49482},
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

    return describe_availability(_availability)


@check_type
def deep_model(model: str = 'vggvox-v2', **kwargs):
    """
    Load language detection deep model.

    Parameters
    ----------
    model : str, optional (default='vggvox-v2')
        Model architecture supported. Allowed values:

        * ``'vggvox-v1'`` - VGGVox V1, embedding size 1024.
        * ``'vggvox-v2'`` - VGGVox V2, embedding size 512.

    Returns
    -------
    result : malaya_speech.model.tf.CLASSIFICATION class
    """
    model = model.lower()
    if model not in _availability:
        raise Exception(
            'model not supported, please check supported models from malaya_speech.language_detection.available_model()'
        )
