from malaya_speech.model.webrtc import WebRTC
from malaya_speech.supervised import classification
from herpetologist import check_type

_availability = {
    'vggvox-v1': {
        'Size (MB)': 70.8,
        'Quantized Size (MB)': 17.7,
        'Accuracy': 0.95,
    },
    'vggvox-v2': {
        'Size (MB)': 31.1,
        'Quantized Size (MB)': 7.92,
        'Accuracy': 0.9594,
    },
    'speakernet': {
        'Size (MB)': 20.3,
        'Quantized Size (MB)': 5.18,
        'Accuracy': 0.9,
    },
}


def available_model():
    """
    List available VAD deep models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(_availability)


@check_type
def webrtc(
    aggressiveness: int = 3,
    sample_rate: int = 16000,
    minimum_amplitude: int = 100,
):
    """
    Load WebRTC VAD model.

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
    except:
        raise ModuleNotFoundError(
            'webrtcvad not installed. Please install it by `pip install webrtcvad` and try again.'
        )

    vad = webrtcvad.Vad(aggressiveness)
    return WebRTC(vad, sample_rate, minimum_amplitude)


@check_type
def deep_model(model: str = 'vggvox-v2', quantized: bool = False, **kwargs):
    """
    Load VAD model.

    Parameters
    ----------
    model : str, optional (default='vggvox-v2')
        Model architecture supported. Allowed values:

        * ``'vggvox-v1'`` - finetuned VGGVox V1.
        * ``'vggvox-v2'`` - finetuned VGGVox V2.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model. 
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.supervised.classification.load function
    """
    model = model.lower()
    if model not in _availability:
        raise Exception(
            'model not supported, please check supported models from `malaya_speech.vad.available_model()`.'
        )

    settings = {
        'vggvox-v1': {'frame_len': 0.005, 'frame_step': 0.0005},
        'vggvox-v2': {'hop_length': 24, 'concat': False, 'mode': 'eval'},
        'speakernet': {'frame_ms': 20, 'stride_ms': 1.0},
    }

    return classification.load(
        model = model,
        module = 'vad',
        extra = settings[model],
        label = [False, True],
        quantized = quantized,
        **kwargs
    )
