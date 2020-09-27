from malaya_speech.model.webrtc import WEBRTC
from malaya_speech.model.frame import FRAME
from malaya_speech.path import PATH_VAD, S3_PATH_VAD
from malaya_speech.supervised import classification
from herpetologist import check_type
from typing import List, Tuple

_availability = {
    'vggvox-v1': {'Size (MB)': 70.8, 'Accuracy': 0},
    'vggvox-v2': {'Size (MB)': 31.1, 'Accuracy': 0.959375},
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
        minimum_amplitude to assume a sample is a voice activity. Else, automatically False.

    Returns
    -------
    result : malaya_speech.model.webrtc.WEBRTC class
    """

    try:
        import webrtcvad
    except:
        raise ValueError(
            'webrtcvad not installed. Please install it by `pip install webrtcvad` and try again.'
        )

    vad = webrtcvad.Vad(aggressiveness)
    return WEBRTC(vad, sample_rate, minimum_amplitude)


@check_type
def deep_model(model: str = 'vggvox-v2', **kwargs):
    """
    Load VAD model.

    Parameters
    ----------
    model : str, optional (default='vggvox-v2')
        Model architecture supported. Allowed values:

        * ``'vggvox-v1'`` - finetuned VGGVox V1.
        * ``'vggvox-v2'`` - finetuned VGGVox V2.

    Returns
    -------
    result : malaya_speech.model.tf.CLASSIFICATION class
    """
    model = model.lower()
    if model not in _availability:
        raise Exception(
            'model not supported, please check supported models from malaya_speech.vad.available_model()'
        )

    settings = {'vggvox-v2': {'hop_length': 24}}

    return classification.load(
        PATH_VAD, S3_PATH_VAD, model, 'vad', settings[model], [False, True]
    )


def group_vad(frames: List[Tuple[FRAME, bool]]):
    """
    Group multiple frames based on label.

    Parameters
    ----------
    frames: List[Tuple[FRAME, bool]]

    Returns
    -------
    result : List[FRAME]
    """
    results, result, last = [], [], None

    for frame in frames:
        if last is None:
            last = frame[1]
            result.append(frame[0])
        elif last == frame[1]:
            result.append(frame[0])
        else:
            a, duration = [], 0
            for r in result:
                a.extend(r.array)
                duration += r.duration
            results.append((FRAME(a, result[0].timestamp, duration), last))
            result = [frame[0]]
            last = frame[1]

    if len(result):
        a, duration = [], 0
        for r in result:
            a.extend(r.array)
            duration += r.duration
        results.append((FRAME(a, result[0].timestamp, duration), last))
    return results
