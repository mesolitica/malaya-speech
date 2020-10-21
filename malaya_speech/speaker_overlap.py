from herpetologist import check_type

_availability = {
    'vggvox-v2': {'Size (MB)': 31.1, 'Accuracy': 0.99},
    'deep-speaker': {'Size (MB)': 30.9, 'Accuracy': 0.99},
}


def available_model():
    """
    List available speaker overlap deep models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(_availability)


@check_type
def deep_model(model: str = 'vggvox-v2', **kwargs):
    """
    Load speaker overlap deep model.

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
            'model not supported, please check supported models from `malaya_speech.speaker_overlap.available_model()`.'
        )
