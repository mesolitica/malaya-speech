import numpy as np
from malaya_speech.model.frame import Frame


def arange_timestamp(features, signal, sr):
    secs = len(signal) / sr
    skip = secs / features.shape[0]
    aranged = [(a, a + skip) for a in np.arange(0, secs, skip)]
    return aranged


def arange_frames(features, signal, sr):
    aranged_timestamp = arange_timestamp(features, signal, sr)
    skip = len(signal) // len(features)
    frames = []
    for i in range(len(aranged_timestamp)):
        frame = Frame(array=signal[i * skip: (i + 1) * skip],
                      timestamp=aranged_timestamp[i][0],
                      duration=aranged_timestamp[i][1] - aranged_timestamp[i][0])
        frames.append(frame)
    return frames
