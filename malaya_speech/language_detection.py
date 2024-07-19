from malaya_speech.supervised import classification

available_nemo = {
    'huseinzol05/nemo-speaker-count-speakernet': {
        'original from': 'https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/vad_marblenet',
        'Size (MB)': 16.2,
    },
    'huseinzol05/nemo-speaker-count-titanet_large': {
        'original from': 'https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/vad_multilingual_marblenet',
        'Size (MB)': 88.8,
    },
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


def nemo(
    model: str = 'huseinzol05/nemo-language-detection-speakernet',
    **kwargs,
):
    """
    Load Nvidia Nemo speaker count model.
    Trained on 200, 300 ms frames.

    Parameters
    ----------
    model : str, optional (default='huseinzol05/nemo-language-detection-speakernet')
        Check available models at `malaya_speech.language_detection.available_nemo()`.

    Returns
    -------
    result : malaya_speech.torch_model.nemo.Classification class
    """
    if model not in _nemo_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.language_detection.available_nemo()`.'
        )

    return classification.nemo_classification(
        model=model,
        label=labels,
        **kwargs
    )
