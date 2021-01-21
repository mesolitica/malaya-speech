import numpy as np
from malaya_speech.utils.astype import float_to_int, to_byte, to_ndarray
from malaya_speech.model.frame import Frame


class WebRTC:

    __name__ = 'vad'

    def __str__(self):
        return f'<{self.__name__}>'

    def __init__(self, vad, sample_rate = 16000, minimum_amplitude: int = 100):
        self.vad = vad
        self.sample_rate = sample_rate
        self.minimum_amplitude = minimum_amplitude

        self.minimum_sample = 30
        self.maximum_sample = 30

    def is_speech(self, frame):

        if isinstance(frame, Frame):
            frame = frame.array

        frame = to_byte(frame)

        minimum = np.mean(np.abs(to_ndarray(frame)))

        return (
            self.vad.is_speech(frame, self.sample_rate)
            and minimum >= self.minimum_amplitude
        )

    def __call__(self, frame):
        return self.is_speech(frame)
