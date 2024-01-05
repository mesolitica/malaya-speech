from malaya_speech.model.webrtc import WebRTC
from malaya_speech.supervised import classification


available_nemo = {
    'huseinzol05/nemo-vad-marblenet': {
        'original from': 'https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/vad_marblenet',
        'Size (MB)': 0.481,
        'macro precision': 0.60044,
        'macro recall': 0.72819,
        'macro f1-score': 0.49225,
    },
    'huseinzol05/nemo-vad-multilingual-marblenet': {
        'original from': 'https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/vad_multilingual_marblenet',
        'Size (MB)': 0.481,
        'macro precision': 0.47633,
        'macro recall': 0.49895,
        'macro f1-score': 0.46915,
    },
}

silero_vad = {
    'original from': 'https://github.com/snakers4/silero-vad',
    'macro precision': 0.60044,
    'macro recall': 0.72819,
    'macro f1-score': 0.49225,
}

webrtc_vad = {
    'notebook link': 'https://github.com/huseinzol05/malaya-speech/blob/master/pretrained-model/vad/evaluate/webrtc.ipynb',
    'macro precision': 0.47163,
    'macro recall': 0.46148,
    'macro f1-score': 0.46461,
}


def webrtc(
    aggressiveness: int = 3,
    sample_rate: int = 16000,
    minimum_amplitude: int = 100,
):
    """
    Load WebRTC VAD model.
    WebRTC prefer 30ms frame, https://github.com/wiseman/py-webrtcvad#how-to-use-it

    Parameters
    ----------
    aggressiveness: int, optional (default=3)
        an integer between 0 and 3.
        0 is the least aggressive about filtering out non-speech, 3 is the most aggressive.
    sample_rate: int, optional (default=16000)
        sample rate for samples.
    minimum_amplitude: int, optional (default=100)
        abs(minimum_amplitude) to assume a sample is a voice activity. Else, automatically False.

    Returns
    -------
    result : malaya_speech.model.webrtc.WebRTC class
    """

    try:
        import webrtcvad
    except BaseException:
        raise ModuleNotFoundError(
            'webrtcvad not installed. Please install it by `pip install webrtcvad` and try again.'
        )

    vad = webrtcvad.Vad(aggressiveness)
    return WebRTC(vad, sample_rate, minimum_amplitude)


def nemo(
    model: str = 'huseinzol05/nemo-vad-marblenet',
    **kwargs,
):
    """
    Load Nemo VAD model.
    Nemo VAD prefer 63 ms frame, https://github.com/NVIDIA/NeMo/blob/02cf155b020964992a974e030b9e318426761e33/nemo/collections/asr/data/feature_to_label_dataset.py#L43

    Parameters
    ----------
    model : str, optional (default='huseinzol05/vad-marblenet')
        Check available models at `malaya_speech.vad.available_nemo`.

    Returns
    -------
    result : malaya_speech.torch_model.nemo.Classification class
    """
    if model not in available_nemo:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.vad.available_nemo`.'
        )

    return classification.nemo_classification(
        model=model,
        label=[False, True],
        **kwargs
    )
