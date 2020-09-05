from malaya_speech.utils.astype import to_byte, to_ndarray
from malaya_speech.model.interface import FRAME
import numpy as np
from herpetologist import check_type


class VAD:
    __name__ = 'vad'


class WEBRTC(VAD):
    def __init__(self, vad, minimum_amplitude: int = 10):
        self.vad = vad
        self.minimum_amplitude = minimum_amplitude

    def is_speech(self, frame, sample_rate):

        if isinstance(frame, FRAME):
            frame = frame.array

        frame = to_byte(frame)

        minimum = np.mean(np.abs(to_ndarray(frame)))

        return (
            self.vad.is_speech(frame, sample_rate)
            and minimum >= self.minimum_amplitude
        )

    def __call__(self, frame, sample_rate):
        return self.is_speech(frame, sample_rate)


@check_type
def webrtc(aggressiveness: int = 3, minimum_amplitude: int = 10):
    try:
        import webrtcvad
    except:
        raise ValueError(
            'webrtcvad not installed. Please install it by `pip install webrtcvad` and try again.'
        )

    vad = webrtcvad.Vad(aggressiveness)
    return WEBRTC(vad, minimum_amplitude)
