from malaya_speech.supervised import classification

available_nemo = {
    'huseinzol05/nemo-is-clean-speakernet': {
        'Size (MB)': 16.2,
        'frame size (MS)': [100, 200, 300],
    },
    'huseinzol05/nemo-is-clean-titanet_large': {
        'Size (MB)': 88.8,
        'frame size (MS)': [100, 200, 300],
    },
}

available_huggingface = {
    'mesolitica/whisper-base-isclean': {
        'Size (MB)': 82.9,
        'frame size (MS)': [600, 800, 1000],
    },
    'mesolitica/whisper-tiny-isclean': {
        'Size (MB)': 82.9,
        'frame size (MS)': [600, 800, 1000],
    }
}


def nemo(
    model: str = 'huseinzol05/nemo-is-clean-speakernet',
    **kwargs,
):
    """
    Load Nvidia Nemo is clean model.
    Trained on 100, 200, 300 ms frames.

    Parameters
    ----------
    model : str, optional (default='huseinzol05/nemo-is-clean-speakernet')
        Check available models at `malaya_speech.is_clean.available_nemo`.

    Returns
    -------
    result : malaya_speech.torch_model.nemo.Classification class
    """
    if model not in available_nemo:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.is_clean.available_nemo`.'
        )

    return classification.nemo_classification(
        model=model,
        label=[False, True],
        **kwargs
    )
